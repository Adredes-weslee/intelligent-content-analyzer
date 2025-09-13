"""LLM generation microservice (Gemini-only with router-based selection).

Enhancements:
- Post-parse heuristics to construct structured citations when model
  output isn't valid JSON:
  - Map bracket indices like [1], [2] to context chunk order.
  - Fallback to top-2 context chunks as citations.

This service generates answers and document summaries using Google Gemini.
It selects between ``gemini-2.5-flash`` (fast) and ``gemini-2.5-pro``
(reasoning) via a lightweight router prompt for QA. Summarization uses the
fast model by default.

Endpoints:
- POST `/generate`: QA answer generation grounded in provided context chunks.
- POST `/summarize`: Summarize provided chunks into a concise overview and
    key insights.

Behavior:
- If OFFLINE_MODE is enabled, both endpoints return deterministic placeholders
    grounded on the first context chunk with basic citations.
- Otherwise:
    - /generate: classify question difficulty (fast vs reasoning) using a router
        prompt and call the selected Gemini model with a grounded system prompt
        that forbids using outside knowledge and mandates citations.
    - /summarize: use the fast Gemini model with a summarizer system prompt to
        produce a concise summary, key points, and citations.

Observability and endpoint contracts remain unchanged.

Env/Settings keys used:
- GEMINI_API_KEY
- Settings.gemini_fast_model (default: gemini-2.5-flash)
- Settings.gemini_reasoning_model (default: gemini-2.5-pro)
"""

from __future__ import annotations

import json
import os
from typing import List

from fastapi import FastAPI

from shared.models import (
    Citation,
    GenerateRequest,
    QARequest,
    QAResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from shared.settings import Settings
from shared.tracing import estimate_tokens, log_event, span

from .prompts import GENERATOR_SYSTEM_PROMPT, ROUTER_PROMPT, SUMMARIZER_SYSTEM_PROMPT

app = FastAPI(title="LLM Generation Service", version="0.2.0")
s = Settings()

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai = None
if _GEMINI_API_KEY and not s.offline_mode:
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=_GEMINI_API_KEY)
    except Exception:
        genai = None


def _route_model(question: str) -> dict:
    """Return {"tier": "fast|reasoning", "model": str, "why": str}."""
    default = {"tier": "fast", "model": s.gemini_fast_model, "why": "default"}
    if genai is None:
        return default
    try:
        router = genai.GenerativeModel(s.gemini_fast_model)
        resp = router.generate_content(
            [
                {"role": "system", "parts": [{"text": ROUTER_PROMPT}]},
                {"role": "user", "parts": [{"text": question}]},
            ]
        )
        text = getattr(resp, "text", "") or ""
        start, end = text.find("{"), text.rfind("}")
        data = {}
        if start != -1 and end != -1:
            data = json.loads(text[start : end + 1])
        tier = (data.get("tier") or "fast").strip().lower()
        why = data.get("why") or ""
        model = s.gemini_reasoning_model if tier == "reasoning" else s.gemini_fast_model
        return {"tier": tier, "model": model, "why": why}
    except Exception:
        return default


def _map_bracket_citations(answer_text: str) -> List[int]:
    """Extract citation indices like [1], [2] from answer text."""
    import re

    idxs: List[int] = []
    for m in re.finditer(r"\[(\d{1,2})\]", answer_text or ""):
        try:
            n = int(m.group(1))
            if n >= 1:
                idxs.append(n - 1)  # zero-based
        except Exception:
            continue
    return list(dict.fromkeys(idxs))  # de-dup, preserve order


@app.post("/generate", response_model=QAResponse)
async def generate_answer(request: GenerateRequest | QARequest) -> QAResponse:
    """Generate an answer using Gemini, grounded in provided context.

    For now, the gateway will compose the context string and include it
    in diagnostics when calling this service (future: pass explicit chunks).
    If Gemini is not available, we return a graceful fallback.
    """
    # Build user prompt. Prefer structured context if provided; otherwise
    # assume the gateway embedded the context into the question string.
    context_str = None
    if hasattr(request, "context_chunks") and isinstance(
        getattr(request, "context_chunks"), list
    ):
        chunks = getattr(request, "context_chunks")
        try:
            context_str = "\n\n".join(
                f"[{i + 1}] doc_id={getattr(c, 'doc_id', '')} page={getattr(getattr(c, 'meta', None), 'page', None)} section={getattr(getattr(c, 'meta', None), 'section', None)}\n{getattr(c, 'text', '')}"
                for i, c in enumerate(chunks)
            )
        except Exception:
            context_str = None
    # The gateway includes the context inline within the question payload when not using structured request
    # Deterministic offline answer
    if s.offline_mode:
        q = getattr(request, "question", "")
        # Use first chunk to generate simple cited response deterministically
        citations: List[Citation] = []
        context = getattr(request, "context_chunks", []) or []
        if context:
            top = context[0]
            answer = f"Based on the document, {q.strip()} -> {top.text[:120]} [Doc {top.doc_id}]"
            citations = [
                Citation(
                    doc_id=getattr(top, "doc_id", ""),
                    page=getattr(top.meta, "page", None),
                    section=getattr(top.meta, "section", None),
                )
            ]
        else:
            answer = f"Insufficient context to answer: {q.strip()}"
        return QAResponse(
            answer=answer,
            citations=citations,
            confidence=0.0,
            diagnostics={"mode": "offline"},
        )

    # Provider routing: Gemini only; fallback if unavailable
    if genai is None:
        q = getattr(request, "question", "")
        answer = f"[fallback] No LLM provider configured. Question: {q[:400]}..."
        return QAResponse(
            answer=answer,
            citations=[],
            confidence=0.0,
            diagnostics={"reason": "no_provider"},
        )

    # Build prompt for Gemini
    system = GENERATOR_SYSTEM_PROMPT
    q = getattr(request, "question", "")
    if context_str:
        user_prompt = (
            f"Question:\n{q}\n\n"
            f"Context chunks (use only these as evidence):\n{context_str}\n\n"
            f"Return JSON as instructed in system prompt."
        )
    else:
        user_prompt = q + "\n\nReturn JSON as instructed in system prompt."
    # Route to fast vs reasoning model
    route = _route_model(q)
    chosen_model = route.get("model") or s.gemini_fast_model
    try:
        model = genai.GenerativeModel(chosen_model)
        corr = getattr(request, "correlation_id", None)
        with span(
            "llm.generate.call",
            model=chosen_model,
            prompt_tokens=estimate_tokens(user_prompt),
            corr=corr,
        ):
            resp = model.generate_content(
                [
                    {"role": "system", "parts": [{"text": system}]},
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ]
            )
        text = resp.text if hasattr(resp, "text") else None
        if text:
            start = text.find("{")
            end = text.rfind("}")
            payload = (
                json.loads(text[start : end + 1]) if start != -1 and end != -1 else {}
            )
            answer = payload.get("answer") or text
            raw_citations = payload.get("citations") or []
            citations: List[Citation] = []
            for c in raw_citations:
                try:
                    citations.append(
                        Citation(
                            doc_id=str(c.get("doc_id")),
                            page=c.get("page"),
                            section=c.get("section"),
                        )
                    )
                except Exception:
                    continue

            # Heuristic citations if none parsed
            if not citations:
                # Try mapping [1], [2] to provided context order
                ctx_chunks = getattr(request, "context_chunks", []) or []
                idxs = _map_bracket_citations(answer)
                for i in idxs:
                    if 0 <= i < len(ctx_chunks):
                        cc = ctx_chunks[i]
                        citations.append(
                            Citation(
                                doc_id=getattr(cc, "doc_id", ""),
                                page=getattr(getattr(cc, "meta", None), "page", None),
                                section=getattr(
                                    getattr(cc, "meta", None), "section", None
                                ),
                            )
                        )
                # Fallback to first two chunks if still empty
                if not citations and ctx_chunks:
                    for cc in ctx_chunks[:2]:
                        citations.append(
                            Citation(
                                doc_id=getattr(cc, "doc_id", ""),
                                page=getattr(getattr(cc, "meta", None), "page", None),
                                section=getattr(
                                    getattr(cc, "meta", None), "section", None
                                ),
                            )
                        )

            out = QAResponse(
                answer=answer,
                citations=citations,
                confidence=0.0,
                diagnostics={
                    "provider": "gemini",
                    "model": chosen_model,
                    "router": {"tier": route.get("tier"), "why": route.get("why")},
                },
            )
            log_event(
                "Generation",
                payload={
                    "type": "qa",
                    "model": chosen_model,
                    "prompt_tokens": estimate_tokens(user_prompt),
                    "output_tokens": estimate_tokens(answer or text),
                },
                correlation_id=corr,
            )
            return out
    except Exception:
        pass

    # Final fallback
    q = getattr(request, "question", "")
    answer = f"[fallback] No provider succeeded. Question: {q[:400]}..."
    return QAResponse(
        answer=answer, citations=[], confidence=0.0, diagnostics={"reason": "fallback"}
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest) -> SummarizeResponse:
    """Summarize a document from provided chunks using Gemini (fast model)."""
    chunks = request.chunks or []
    # Offline mode: deterministic summary from first chunk
    if s.offline_mode or genai is None:
        text = getattr(chunks[0], "text", "") if chunks else ""
        summary = f"Summary: {text[:200]}..." if text else "No content to summarize."
        citations: List[Citation] = []
        if chunks:
            c0 = chunks[0]
            citations.append(
                Citation(
                    doc_id=getattr(c0, "doc_id", request.doc_id or ""),
                    page=getattr(getattr(c0, "meta", None), "page", None),
                    section=getattr(getattr(c0, "meta", None), "section", None),
                )
            )
        return SummarizeResponse(
            summary=summary,
            key_points=[] if not text else [text[:80] + "..."],
            citations=citations,
            diagnostics={"mode": "offline"},
        )

    # Build summarization prompt
    try:
        context_str = "\n\n".join(
            [
                f"[{i + 1}] doc_id={getattr(c, 'doc_id', request.doc_id or '')} "
                f"page={getattr(getattr(c, 'meta', None), 'page', None)} "
                f"section={getattr(getattr(c, 'meta', None), 'section', None)}\n"
                f"{getattr(c, 'text', '')}"
                for i, c in enumerate(chunks)
            ]
        )
    except Exception:
        context_str = "\n\n".join([getattr(c, "text", "") for c in chunks])

    try:
        model = genai.GenerativeModel(s.gemini_fast_model)
        resp = model.generate_content(
            [
                {"role": "system", "parts": [{"text": SUMMARIZER_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Summarize the following chunks. "
                                "Return JSON as instructed in the system prompt.\n\n"
                                f"{context_str}"
                            )
                        }
                    ],
                },
            ]
        )
        text = resp.text if hasattr(resp, "text") else None
        if text:
            start = text.find("{")
            end = text.rfind("}")
            try:
                payload = (
                    json.loads(text[start : end + 1])
                    if start != -1 and end != -1
                    else {}
                )
            except Exception:
                payload = {"summary": text, "key_points": []}
            summary = payload.get("summary") or ""
            key_points = payload.get("key_points") or []
            raw_citations = payload.get("citations") or []
            citations: List[Citation] = []
            for c in raw_citations:
                try:
                    citations.append(
                        Citation(
                            doc_id=str(c.get("doc_id")),
                            page=c.get("page"),
                            section=c.get("section"),
                        )
                    )
                except Exception:
                    continue
            return SummarizeResponse(
                summary=summary,
                key_points=key_points if isinstance(key_points, list) else [],
                citations=citations,
                diagnostics={"provider": "gemini", "model": s.gemini_fast_model},
            )
    except Exception:
        pass

    # Fallback
    return SummarizeResponse(
        summary="Unable to summarize at this time.",
        key_points=[],
        citations=[],
        diagnostics={"reason": "fallback"},
    )
