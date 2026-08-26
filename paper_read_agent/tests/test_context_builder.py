from types import SimpleNamespace

from paper_read_agent.domain.models import Chunk
from paper_read_agent.retrieval.context_builder import ContextBuilder
from paper_read_agent.retrieval.hybrid import HybridCandidate
from paper_read_agent.retrieval.reranker import RerankedCandidate


class Chars:
    def encode(self, text, *, add_special_tokens=False): return list(text)


def chunk(cid, text, parent=None, kind="text", quality=.9):
    return Chunk(cid, "v", parent, text, len(text), 1, 1, (), kind, quality_score=quality)


def ranked(cid, text, paper="p1", matches=("q",)):
    return RerankedCandidate(HybridCandidate(cid, text, .1, {"paper_id": paper}, ()), .9, matches)


def test_expands_parent_neighbors_and_special_relations_with_traceability():
    chunks = {x.chunk_id: x for x in [chunk("s", "seed", "p"), chunk("p", "parent"),
                                      chunk("n", "neighbor"), chunk("f", "formula", kind="formula")]}
    result = ContextBuilder(Chars()).build([ranked("s", "seed")], chunks=chunks,
        paper_ids={k: "paper" for k in chunks}, adjacency={"s": (None, "n")},
        special_relations={"s": ["f"]}, model_window_tokens=100)
    assert [x.reason for x in result.items] == ["reranked", "parent", "adjacent", "special_relation"]
    assert all(x.seed_chunk_ids == ("s",) for x in result.items)


def test_deduplicates_and_excludes_low_quality_content():
    chunks = {"a": chunk("a", "same"), "b": chunk("b", " same "), "bad": chunk("bad", "bad", quality=.1)}
    result = ContextBuilder(Chars()).build([ranked("a", "same"), ranked("b", "same"), ranked("bad", "bad")],
        chunks=chunks, paper_ids={k: "p" for k in chunks}, adjacency={}, model_window_tokens=100)
    assert [x.chunk.chunk_id for x in result.items] == ["a"]
    assert "bad" in result.omitted_chunk_ids


def test_small_window_never_exceeds_budget_and_reports_omissions():
    chunks = {"a": chunk("a", "123456"), "b": chunk("b", "abcdef")}
    result = ContextBuilder(Chars(), evidence_ratio=.5).build([ranked("a", ""), ranked("b", "")],
        chunks=chunks, paper_ids={"a": "p1", "b": "p2"}, adjacency={}, model_window_tokens=20)
    assert result.token_budget == 10 and result.token_count <= 10
    assert len(result.items) == 1 and result.omitted_chunk_ids == ("b",)


def test_same_content_from_different_papers_is_kept_for_comparison():
    chunks = {"a": chunk("a", "same"), "b": chunk("b", "same")}
    result = ContextBuilder(Chars()).build([ranked("a", "", "p1"), ranked("b", "", "p2")],
        chunks=chunks, paper_ids={"a": "p1", "b": "p2"}, adjacency={}, model_window_tokens=100)
    assert len(result.items) == 2


def test_invalid_budget_ratio_is_rejected():
    try: ContextBuilder(Chars(), evidence_ratio=.7)
    except ValueError as exc: assert "between 0.4 and 0.6" in str(exc)
    else: raise AssertionError("ratio must fail")
