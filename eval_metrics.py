"""Evaluation metrics for the PDF retrieval-augmented generation pipeline.

The retrieval metrics are deterministic.  The answer-quality metrics use an
injectable judge function so this module can be tested without loading an LLM.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "with",
}


def _key(item: Any) -> str:
    if isinstance(item, dict):
        source = item.get("source", "")
        page = item.get("page", "")
        chunk = item.get("chunk", "")
        return f"{source}|{page}|{chunk}".lower().strip()
    return str(item).lower().strip()


def _matches(retrieved: Any, relevant: Any) -> bool:
    retrieved_key = _key(retrieved)
    relevant_key = _key(relevant)
    if retrieved_key == relevant_key:
        return True

    # Allow a source-level ground truth such as "guide.pdf" to match any
    # retrieved chunk from that document.
    if isinstance(relevant, str) and "|" not in relevant_key:
        return retrieved_key.split("|", 1)[0] == relevant_key
    return False


def hit_rate_at_k(retrieved: Sequence[Any], relevant: Iterable[Any], k: int = 5) -> float:
    """Return 1 when at least one relevant item appears in the top-k results."""
    relevant_items = list(relevant)
    return float(any(_matches(item, truth) for item in retrieved[:k] for truth in relevant_items))


def reciprocal_rank(retrieved: Sequence[Any], relevant: Iterable[Any]) -> float:
    """Return the reciprocal rank of the first relevant retrieved item."""
    relevant_items = list(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if any(_matches(item, truth) for truth in relevant_items):
            return 1.0 / rank
    return 0.0


def retrieval_metrics(retrieved: Sequence[Any], relevant: Iterable[Any], k: int = 5) -> dict[str, float]:
    return {
        f"Hit Rate@{k}": hit_rate_at_k(retrieved, relevant, k),
        "MRR": reciprocal_rank(retrieved, relevant),
    }


def _context(matches: Sequence[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[Source {i}] Document: {item.get('source')} | Page: {item.get('page')} | "
        f"Chunk: {item.get('chunk')}\n{item.get('text', '')}"
        for i, item in enumerate(matches, start=1)
    )


def _terms(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOP_WORDS and len(word) > 2
    }


def _overlap(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def fast_answer_metrics(
    question: str,
    answer: str,
    matches: Sequence[dict[str, Any]],
    reference_answer: str = "",
) -> dict[str, float]:
    """Fast local approximation of answer metrics; does not call an LLM."""
    clean_answer = re.sub(r"\[Source\s+\d+\]", "", answer, flags=re.IGNORECASE)
    context_by_source = {
        i: item.get("text", "") for i, item in enumerate(matches, start=1)
    }
    context = "\n".join(context_by_source.values())
    claims = [part.strip() for part in re.split(r"[.!?\n]+", clean_answer) if part.strip()]

    faithfulness = sum(_overlap(claim, context) for claim in claims) / max(len(claims), 1)
    if reference_answer:
        answer_correctness = (_overlap(answer, reference_answer) + _overlap(reference_answer, answer)) / 2
    else:
        answer_correctness = faithfulness

    citations = [int(number) for number in re.findall(r"\[Source\s+(\d+)\]", answer, re.IGNORECASE)]
    if not citations:
        citation_accuracy = 0.0
    else:
        valid = [number for number in citations if number in context_by_source]
        citation_accuracy = len(valid) / len(citations)

    return {
        "Faithfulness": round(max(0.0, min(1.0, faithfulness)), 3),
        "Answer Correctness": round(max(0.0, min(1.0, answer_correctness)), 3),
        "Citation Accuracy": round(max(0.0, min(1.0, citation_accuracy)), 3),
    }


def _parse_scores(raw: str) -> dict[str, float]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("The evaluation judge did not return JSON.")
    values = json.loads(match.group(0))
    result = {}
    for name in ("Faithfulness", "Answer Correctness", "Citation Accuracy"):
        result[name] = max(0.0, min(1.0, float(values[name])))
    return result


def answer_metrics(
    question: str,
    answer: str,
    matches: Sequence[dict[str, Any]],
    reference_answer: str = "",
    judge: Callable[[str], str] | None = None,
) -> dict[str, float]:
    """Score answer quality from 0 to 1 using a supplied LLM judge."""
    if not answer or not matches:
        return {"Faithfulness": 0.0, "Answer Correctness": 0.0, "Citation Accuracy": 0.0}
    if judge is None:
        raise ValueError("An LLM judge is required for answer metrics.")

    prompt = f"""
You are an evaluation judge. Score the answer from 0 to 1 for each metric.
Faithfulness: every factual claim is supported by the supplied context.
Answer Correctness: the answer correctly addresses the question and agrees with
the reference answer when one is supplied.
Citation Accuracy: inline [Source N] citations point to sources that support the
claims they cite. If there are no inline citations, score 0.
Return JSON only with exactly these numeric keys: Faithfulness,
Answer Correctness, Citation Accuracy.

Question: {question}
Reference answer: {reference_answer or "(none supplied)"}
Answer: {answer}
Context:
{_context(matches)}
"""
    return _parse_scores(judge(prompt))


def format_metrics(metrics: dict[str, float]) -> str:
    return "\n".join(f"**{name}:** {value:.3f}" for name, value in metrics.items())
