import pytest

from eval_metrics import hit_rate_at_k, reciprocal_rank, retrieval_metrics


def test_retrieval_metrics_use_top_five_and_first_relevant_rank():
    retrieved = ["a.pdf", "b.pdf", "target.pdf", "c.pdf", "d.pdf", "target.pdf"]
    assert hit_rate_at_k(retrieved, ["target.pdf"], k=5) == 1.0
    assert reciprocal_rank(retrieved, ["target.pdf"]) == pytest.approx(1 / 3)


def test_retrieval_metrics_return_zero_without_relevant_result():
    assert retrieval_metrics(["a.pdf"], ["missing.pdf"]) == {
        "Hit Rate@5": 0.0,
        "MRR": 0.0,
    }


def test_source_level_truth_matches_a_chunk():
    retrieved = [{"source": "guide.pdf", "page": 2, "chunk": 1}]
    assert hit_rate_at_k(retrieved, ["guide.pdf"]) == 1.0
