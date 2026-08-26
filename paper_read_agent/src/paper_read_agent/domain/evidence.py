"""Trusted evidence objects and citation resolution."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

from paper_read_agent.retrieval.context_builder import EvidenceContext


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    chunk_id: str
    paper_id: str
    paper_title: str
    version_id: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    excerpt: str
    quality_score: float | None
    source_type: str


@dataclass(frozen=True, slots=True)
class Citation:
    evidence_id: str
    label: str
    excerpt: str
    source_type: str


class EvidenceRegistry:
    def __init__(self, evidence: Sequence[Evidence], *, allowed_paper_ids: Sequence[str]) -> None:
        self._items = {item.evidence_id: item for item in evidence}
        self.allowed_paper_ids = frozenset(allowed_paper_ids)

    @classmethod
    def from_context(
        cls, context: EvidenceContext, *, paper_titles: Mapping[str, str],
        allowed_paper_ids: Sequence[str], source_types: Mapping[str, str] | None = None,
    ) -> "EvidenceRegistry":
        source_types = source_types or {}
        items = []
        for item in context.items:
            chunk = item.chunk
            evidence_id = "ev_" + sha256(
                f"{chunk.version_id}\x1f{chunk.chunk_id}\x1f{chunk.page_start}\x1f{chunk.page_end}".encode()
            ).hexdigest()[:20]
            items.append(Evidence(
                evidence_id, chunk.chunk_id, item.paper_id,
                paper_titles.get(item.paper_id, item.paper_id), chunk.version_id,
                chunk.page_start, chunk.page_end, chunk.section_path, chunk.text,
                chunk.quality_score, source_types.get(chunk.chunk_id, "native_pdf"),
            ))
        return cls(items, allowed_paper_ids=allowed_paper_ids)

    @property
    def evidence(self) -> tuple[Evidence, ...]: return tuple(self._items.values())

    def resolve(self, evidence_ids: Sequence[str]) -> tuple[Citation, ...]:
        citations = []
        for evidence_id in dict.fromkeys(evidence_ids):
            item = self._items.get(evidence_id)
            if item is None: raise ValueError(f"Unknown Evidence ID: {evidence_id}")
            if item.paper_id not in self.allowed_paper_ids:
                raise ValueError(f"Evidence is outside the active paper scope: {evidence_id}")
            pages = f"p. {item.page_start}" if item.page_start == item.page_end else f"pp. {item.page_start}-{item.page_end}"
            section = " > ".join(item.section_path) or "Unsectioned"
            citations.append(Citation(
                evidence_id, f"{item.paper_title} ({item.version_id}), {pages}, {section}",
                item.excerpt, item.source_type,
            ))
        return tuple(citations)
