import pytest

from paper_read_agent.domain.evidence import EvidenceRegistry
from paper_read_agent.domain.models import Chunk
from paper_read_agent.retrieval.context_builder import ContextItem, EvidenceContext


def context(source="native_pdf"):
    chunks = [Chunk("c1", "v1", None, "method text", 2, 2, 3, ("Methods",), "text", .9),
              Chunk("c2", "v2", None, "ocr text", 2, 5, 5, (), "text", .7)]
    ctx = EvidenceContext(tuple(ContextItem(c, "p1", "reranked", (c.chunk_id,)) for c in chunks), 4, 10, ())
    return ctx


def test_stable_single_and_multiple_evidence_citations():
    first = EvidenceRegistry.from_context(context(), paper_titles={"p1": "Same title"},
                                          allowed_paper_ids=["p1"], source_types={"c2": "rapidocr"})
    second = EvidenceRegistry.from_context(context(), paper_titles={"p1": "Same title"},
                                           allowed_paper_ids=["p1"])
    assert [x.evidence_id for x in first.evidence] == [x.evidence_id for x in second.evidence]
    cites = first.resolve([x.evidence_id for x in first.evidence])
    assert "pp. 2-3" in cites[0].label and "v1" in cites[0].label
    assert cites[1].source_type == "rapidocr" and "p. 5" in cites[1].label


def test_invalid_and_out_of_scope_ids_are_rejected():
    registry = EvidenceRegistry.from_context(context(), paper_titles={}, allowed_paper_ids=[])
    with pytest.raises(ValueError, match="Unknown"): registry.resolve(["ev_fake"])
    with pytest.raises(ValueError, match="outside"): registry.resolve([registry.evidence[0].evidence_id])


def test_duplicate_ids_resolve_once_and_metadata_cannot_be_overridden():
    registry = EvidenceRegistry.from_context(context(), paper_titles={"p1": "Paper"}, allowed_paper_ids=["p1"])
    eid = registry.evidence[0].evidence_id
    cites = registry.resolve([eid, eid])
    assert len(cites) == 1 and "p. 99" not in cites[0].label
    assert "pp. 2-3" in cites[0].label
