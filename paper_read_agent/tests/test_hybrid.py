from types import SimpleNamespace

import pytest

from paper_read_agent.retrieval.hybrid import HybridRetriever
from paper_read_agent.retrieval.query_planner import QueryIntent, RetrievalPlan


def hit(chunk_id, text=None, *, paper="p1", kind="text"):
    return SimpleNamespace(chunk_id=chunk_id, text=text or chunk_id,
                           metadata={"paper_id": paper, "content_type": kind})


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = results or {}
        self.error = error
        self.calls = []

    def query(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if self.error:
            raise self.error
        return self.results.get(text, ())


def plan(*, queries=("q",), papers=("p1",), references=False, clarification=False):
    return RetrievalPlan("q", "q", QueryIntent.FACTUAL, tuple(queries), tuple(papers),
                         ("text",), references, clarification)


def test_fixed_rankings_use_rrf_and_keep_source_traces() -> None:
    vector = FakeRetriever({"q": [hit("a"), hit("b")]})
    keyword = FakeRetriever({"q": [hit("b"), hit("c")]})
    result = HybridRetriever(vector, keyword, candidate_limit=30).retrieve(plan())
    assert [item.chunk_id for item in result.candidates] == ["b", "a", "c"]
    assert result.candidates[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert {trace.route for trace in result.candidates[0].traces} == {"vector", "keyword"}


def test_multiple_subqueries_are_deduplicated_and_scope_is_forwarded() -> None:
    vector = FakeRetriever({"q": [hit("a")], "sub": [hit("b")]})
    keyword = FakeRetriever({"q": [hit("a")], "sub": [hit("b")]})
    result = HybridRetriever(vector, keyword, candidate_limit=30).retrieve(
        plan(queries=("q", "sub", "sub"), papers=("p1", "p2"), references=True)
    )
    assert {item.chunk_id for item in result.candidates} == {"a", "b"}
    assert len(vector.calls) == 2
    assert vector.calls[0][1]["paper_ids"] == ["p1", "p2"]
    assert vector.calls[0][1]["include_references"] is True


def test_duplicate_content_is_removed_only_within_same_paper() -> None:
    vector = FakeRetriever({"q": [hit("a", "same", paper="p1"), hit("b", " same ", paper="p1"),
                                   hit("c", "same", paper="p2")]})
    result = HybridRetriever(vector, FakeRetriever(), candidate_limit=30).retrieve(plan())
    assert [item.chunk_id for item in result.candidates] == ["a", "c"]


def test_single_route_failure_degrades_but_keeps_other_results() -> None:
    vector = FakeRetriever(error=RuntimeError("vector unavailable"))
    keyword = FakeRetriever({"q": [hit("k")]})
    result = HybridRetriever(vector, keyword, candidate_limit=30).retrieve(plan())
    assert [item.chunk_id for item in result.candidates] == ["k"]
    assert result.degraded is True
    assert "vector unavailable" in result.route_failures[0]


def test_both_routes_failure_and_empty_results_are_observable() -> None:
    failed = HybridRetriever(FakeRetriever(error=ValueError("v")),
                             FakeRetriever(error=ValueError("k")), candidate_limit=30).retrieve(plan())
    empty = HybridRetriever(FakeRetriever(), FakeRetriever(), candidate_limit=30).retrieve(plan())
    assert failed.candidates == () and failed.degraded is True and len(failed.route_failures) == 2
    assert empty.candidates == () and empty.degraded is False and empty.route_failures == ()


def test_clarification_plan_does_not_retrieve() -> None:
    vector, keyword = FakeRetriever(), FakeRetriever()
    result = HybridRetriever(vector, keyword, candidate_limit=30).retrieve(plan(clarification=True))
    assert result.candidates == ()
    assert vector.calls == keyword.calls == []


def test_empty_hard_scope_never_expands_to_all_documents() -> None:
    vector, keyword = FakeRetriever({"q": [hit("leak")]}), FakeRetriever()
    result = HybridRetriever(vector, keyword, candidate_limit=30).retrieve(plan(papers=()))
    assert result.candidates == ()
    assert vector.calls == keyword.calls == []


def test_candidate_limit_contract() -> None:
    with pytest.raises(ValueError, match="between 30 and 50"):
        HybridRetriever(FakeRetriever(), FakeRetriever(), candidate_limit=20)
