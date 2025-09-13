"""Evaluation microservice.

Provides:
- POST /evaluate: heuristic scores for factuality, relevance, completeness,
    and extended judge rubric: faithfulness, answer_relevance_1_5, and
    context_relevance_ratio. When LLM judging is enabled, enrich metrics via
    Gemini using a strict fact-checker prompt. All scores except
    answer_relevance_1_5 are normalized to [0,1].
- POST /judge: optional LLM-as-judge grading endpoint returning raw judge
    payload. Intended for diagnostics; /evaluate is the canonical API.

This service holds no state and performs no persistence.
"""

from __future__ import annotations

from fastapi import FastAPI

from services.llm_generate.app.prompts import JUDGE_SYSTEM_PROMPT
from shared.models import EvaluateRequest, EvaluateResponse
from shared.settings import Settings
from shared.tracing import install_fastapi_tracing, span

from .metrics import (
    heuristic_answer_relevance_1_5,
    heuristic_context_relevance_ratio,
    heuristic_faithfulness,
    simple_metric_scores,
)

app = FastAPI(title="Evaluation Service", version="0.2.0")
install_fastapi_tracing(app, service_name="evaluation")

_settings = Settings()

_genai = None
if _settings.eval_llm_enabled and not _settings.offline_mode:
    api_key = _settings.gemini_api_key
    if api_key:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            _genai = genai
        except Exception:
            _genai = None


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Compute quality metrics for a generated answer with optional LLM judge.

    Always computes deterministic heuristics. If LLM judging is enabled and
    a Gemini API key is configured, enrich metrics using the judge prompt.
    Logs inputs and outputs via tracing spans when Langfuse is enabled.
    """
    with span(
        "eval.compute",
        question=req.question,
        have_hits=bool(req.hits),
        num_sources=len(req.sources),
    ):
        factuality, relevance, completeness = simple_metric_scores(
            req.question, req.answer, req.sources
        )

        # Extended heuristics
        faithfulness = heuristic_faithfulness(req.answer, req.sources)
        answer_rel_1_5 = heuristic_answer_relevance_1_5(req.question, req.answer)
        ctx_ratio = (
            heuristic_context_relevance_ratio(
                req.question, [h.chunk for h in (req.hits or [])]
            )
            if req.hits
            else None
        )

        # Optional LLM-as-judge enrichment
        judge_payload = None
        # Only call LLM judge if globally enabled AND not explicitly disabled in request
        if _genai is not None and (req.use_judge is not False):
            try:
                text_sources = "\n\n".join(
                    [getattr(c, "text", "") for c in req.sources]
                )
                prompt = (
                    f"Question:\n{req.question}\n\nAnswer:\n{req.answer}\n\nSources:\n{text_sources}\n\n"
                    "Return JSON as instructed in the system prompt."
                )
                with span("eval.judge.llm"):
                    model = _genai.GenerativeModel(
                        _settings.gemini_reasoning_model,
                        system_instruction=JUDGE_SYSTEM_PROMPT,
                    )
                    resp = model.generate_content(prompt)
                    raw = getattr(resp, "text", "") or ""
                    start, end = raw.find("{"), raw.rfind("}")
                    if start != -1 and end != -1:
                        import json as _json

                        judge_payload = _json.loads(raw[start : end + 1])
            except Exception:
                judge_payload = None

        # Map judge outputs if present
        if judge_payload:
            try:
                fs = float(judge_payload.get("factual_score_0_10", 0.0))
                factuality = max(0.0, min(1.0, fs / 10.0))
                if "faithfulness_percent_0_100" in judge_payload:
                    faithfulness = max(
                        0.0,
                        min(
                            1.0,
                            float(judge_payload["faithfulness_percent_0_100"]) / 100.0,
                        ),
                    )
                if "answer_relevance_1_5" in judge_payload:
                    answer_rel_1_5 = float(
                        judge_payload["answer_relevance_1_5"]
                    )  # keep 1–5 scale
                if "context_relevance_ratio_0_1" in judge_payload:
                    ctx_ratio = float(judge_payload["context_relevance_ratio_0_1"])
            except Exception:
                pass

        with span(
            "eval.result",
            factuality=round(factuality, 4),
            relevance=round(relevance, 4),
            completeness=round(completeness, 4),
            faithfulness=round(faithfulness or 0.0, 4),
            answer_relevance_1_5=round(answer_rel_1_5 or 0.0, 3),
            context_relevance_ratio=round(ctx_ratio or 0.0, 4)
            if ctx_ratio is not None
            else None,
        ):
            return EvaluateResponse(
                factuality=factuality,
                relevance=relevance,
                completeness=completeness,
                faithfulness=faithfulness,
                answer_relevance_1_5=answer_rel_1_5,
                context_relevance_ratio=ctx_ratio,
                comments=(
                    "Heuristic with optional LLM-judge enrichment."
                    if judge_payload or _genai
                    else "Heuristic"
                ),
            )


@app.post("/judge")
async def judge(req: EvaluateRequest) -> dict:
    """LLM-as-judge grading (optional). Returns factual_score_0_10 and justification.

    Falls back to a deterministic stub if disabled or unavailable.
    """
    # Optional LLM path
    if _genai is not None:
        try:
            text_sources = "\n\n".join([getattr(c, "text", "") for c in req.sources])
            prompt = (
                f"Question:\n{req.question}\n\nAnswer:\n{req.answer}\n\nSources:\n{text_sources}\n\n"
                "Return JSON as instructed in the system prompt."
            )
            with span("judge.llm.call"):
                model = _genai.GenerativeModel(
                    _settings.gemini_reasoning_model,
                    system_instruction=JUDGE_SYSTEM_PROMPT,
                )
                resp = model.generate_content(prompt)
                raw = getattr(resp, "text", "") or ""
                start, end = raw.find("{"), raw.rfind("}")
                payload = {}
                if start != -1 and end != -1:
                    import json as _json

                    payload = _json.loads(raw[start : end + 1])
                score = float(payload.get("factual_score_0_10", 0.0))
                result = {
                    "factuality": max(0.0, min(1.0, score / 10.0)),
                    "relevance": None,
                    "completeness": None,
                    "justification": payload.get("justification"),
                    "raw": payload,
                }
            with span(
                "judge.llm.result", **{k: v for k, v in result.items() if k != "raw"}
            ):
                return result
        except Exception:
            pass

    # Fallback stub
    factuality, relevance, completeness = _fallback_judge(req)
    return {
        "factuality": factuality,
        "relevance": relevance,
        "completeness": completeness,
    }


# Local fallback judge
def _fallback_judge(req: EvaluateRequest) -> tuple[float, float, float]:
    # Dummy mid-range scores
    return 0.5, 0.5, 0.5
