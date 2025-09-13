"""Heuristic metric functions for answer evaluation.

These functions compute very simple metrics based on substring
containment. They are not robust but are sufficient to demonstrate
the evaluation API. In a real system you would replace these with
proper implementations leveraging natural language inference models
and retrieval precision/recall calculations.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from shared.models import DocChunk

_ABSTAIN_PATTERNS = [
    r"\bnot\s+sure\b",
    r"\bnot\s+provided\b",
    r"\bno\s+(information|context)\b",
    r"\b(can|cannot|can't)\s+find\b",
    r"\binsufficient\s+context\b",
    r"\bdo\s+not\s+know\b",
    r"\bunable\s+to\s+(locate|answer)\b",
]


def _looks_uncertain(answer: str) -> bool:
    a = (answer or "").lower()
    return any(re.search(p, a) for p in _ABSTAIN_PATTERNS)


def _norm_tokens(s: str) -> List[str]:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return [t for t in s.split() if len(t) > 2]


def simple_metric_scores(
    question: str, answer: str, sources: List[DocChunk]
) -> Tuple[float, float, float]:
    """Return (factuality, relevance, completeness) in [0,1]."""
    q = set(_norm_tokens(question))
    a = set(_norm_tokens(answer))
    s = set(_norm_tokens(" ".join(getattr(c, "text", "") for c in (sources or []))))

    abstain = _looks_uncertain(answer)

    # Factuality: answer tokens supported by sources
    factuality = (len(a & s) / max(1, len(a))) if a and s else 0.0

    # Relevance: overlap between question and answer
    relevance = (len(q & a) / max(1, len(q))) if q and a else 0.0

    # Completeness: question tokens covered by answer+sources
    union_as = a | s
    completeness = (len(q & union_as) / max(1, len(q))) if q else 0.0

    if abstain:
        # Don’t reward abstentions with high metrics
        completeness = 0.0
        relevance = relevance * 0.25
        factuality = min(factuality, 0.3)

    return float(factuality), float(relevance), float(completeness)


def _sentences(text: str) -> List[str]:
    """Very naive sentence splitter to keep dependencies light."""
    parts = []
    for sep in [". ", "? ", "! "]:
        if sep in text:
            if not parts:
                parts = text.split(sep)
            else:
                _tmp: List[str] = []
                for p in parts:
                    _tmp.extend(p.split(sep))
                parts = _tmp
    return [p.strip().strip(".?!") for p in (parts or [text]) if p.strip()]


def heuristic_faithfulness(answer: str, sources: List[DocChunk]) -> float:
    """Estimate faithfulness as fraction of answer sentences whose keywords appear in sources.

    This is a crude proxy: for each sentence, build a set of lowercase tokens and
    check if at least half of the tokens appear in the concatenated sources.
    Returns a value in [0,1].
    """
    sents = _sentences(answer)
    if not sents:
        return 0.0
    combined = " ".join(c.text.lower() for c in sources)
    combined_terms = set(combined.split())
    supported = 0
    for s in sents:
        terms = [t for t in s.lower().split() if len(t) > 2]
        if not terms:
            continue
        overlap = sum(1 for t in terms if t in combined_terms)
        if overlap >= max(1, len(terms) // 2):
            supported += 1
    return supported / max(1, len(sents))


def heuristic_answer_relevance_1_5(question: str, answer: str) -> float:
    """Score how directly the answer addresses the question on a 1–5 scale.

    Heuristic: compute token overlap ratio between question and answer; map to 1–5.
    """
    q = set(question.lower().split())
    a = set(answer.lower().split())
    if not a:
        return 1.0
    overlap = len(q & a) / max(1, len(q))
    return 1.0 + 4.0 * max(0.0, min(1.0, overlap))


def heuristic_context_relevance_ratio(question: str, hits: Iterable[DocChunk]) -> float:
    """Estimate ratio of relevant chunks to the question.

    Mark a chunk relevant if at least one question token appears in the chunk text.
    Returns a value in [0,1].
    """
    q_terms = set(t for t in question.lower().split() if len(t) > 2)
    hit_list = list(hits)
    if not hit_list:
        return 0.0
    relevant = 0
    for c in hit_list:
        text = getattr(c, "text", "").lower()
        if any(t in text for t in q_terms):
            relevant += 1
    return relevant / len(hit_list)
