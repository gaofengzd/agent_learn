from paper_read_agent.retrieval.hybrid import HybridCandidate
from paper_read_agent.retrieval.query_planner import QueryIntent, RetrievalPlan
from paper_read_agent.retrieval.reranker import CandidateReranker, LocalBGEReranker


class FixedScorer:
    def __init__(self, values=None, error=None): self.values, self.error = values or {}, error
    def score(self, query, texts):
        if self.error: raise self.error
        return [self.values.get((query, text), 0.0) for text in texts]


def candidate(cid, text, paper="p1"):
    return HybridCandidate(cid, text, 0.01, {"paper_id": paper}, ())


def plan(intent=QueryIntent.FACTUAL, subs=("q",), papers=("p1",)):
    return RetrievalPlan("q", "q", intent, subs, papers, (), False)


def test_ranks_fixed_chinese_and_english_candidates_and_drops_low_scores():
    items = [candidate("a", "研究方法"), candidate("b", "experimental method"), candidate("c", "noise")]
    scorer = FixedScorer({("q", "研究方法"): .8, ("q", "experimental method"): .9,
                          ("q", "noise"): .01})
    result = CandidateReranker(scorer, result_limit=8).rerank(plan(), items)
    assert [x.candidate.chunk_id for x in result.candidates] == ["b", "a"]


def test_multiple_subquestions_use_best_score_and_record_coverage():
    items = [candidate("a", "method"), candidate("b", "result")]
    scorer = FixedScorer({("q", "method"): .2, ("method?", "method"): .9,
                          ("result?", "result"): .8})
    result = CandidateReranker(scorer, result_limit=8).rerank(plan(subs=("method?", "result?")), items)
    assert {x.candidate.chunk_id for x in result.candidates} == {"a", "b"}
    assert "method?" in result.candidates[0].matched_subqueries


def test_comparison_keeps_relevant_multiple_papers_without_padding():
    items = [candidate("a", "a", "p1"), candidate("b", "b", "p2"), candidate("c", "c", "p3")]
    scorer = FixedScorer({("q", "a"): .9, ("q", "b"): .7, ("q", "c"): .01})
    result = CandidateReranker(scorer, result_limit=8).rerank(
        plan(QueryIntent.COMPARISON, papers=("p1", "p2", "p3")), items)
    assert [x.candidate.metadata["paper_id"] for x in result.candidates] == ["p1", "p2"]


def test_failure_returns_explicit_degraded_fallback_and_empty_is_not_degraded():
    reranker = CandidateReranker(FixedScorer(error=RuntimeError("model unavailable")), result_limit=8)
    failed = reranker.rerank(plan(), [candidate("a", "a")])
    empty = reranker.rerank(plan(), [])
    assert failed.degraded and "model unavailable" in failed.error
    assert [x.candidate.chunk_id for x in failed.candidates] == ["a"]
    assert empty.candidates == () and not empty.degraded


def test_missing_local_model_is_explicit(tmp_path):
    try: LocalBGEReranker(tmp_path / "missing")
    except FileNotFoundError as exc: assert "Local reranker" in str(exc)
    else: raise AssertionError("missing model must fail")
