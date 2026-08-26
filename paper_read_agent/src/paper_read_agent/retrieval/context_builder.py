"""Traceable evidence context expansion and token budgeting."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Protocol, Sequence

from paper_read_agent.domain.models import Chunk
from paper_read_agent.retrieval.reranker import RerankedCandidate


class TokenCounter(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class ContextItem:
    chunk: Chunk
    paper_id: str
    reason: str
    seed_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    items: tuple[ContextItem, ...]
    token_count: int
    token_budget: int
    omitted_chunk_ids: tuple[str, ...]


class ContextBuilder:
    def __init__(self, tokenizer: TokenCounter, *, evidence_ratio: float = 0.5,
                 min_quality: float = 0.5) -> None:
        if not 0.4 <= evidence_ratio <= 0.6:
            raise ValueError("Evidence ratio must be between 0.4 and 0.6")
        self.tokenizer, self.evidence_ratio, self.min_quality = tokenizer, evidence_ratio, min_quality

    def build(
        self, ranked: Sequence[RerankedCandidate], *, chunks: Mapping[str, Chunk],
        paper_ids: Mapping[str, str], adjacency: Mapping[str, tuple[str | None, str | None]],
        special_relations: Mapping[str, Sequence[str]] | None = None,
        model_window_tokens: int,
    ) -> EvidenceContext:
        budget = int(model_window_tokens * self.evidence_ratio)
        if budget <= 0: raise ValueError("Model window must be positive")
        special_relations = special_relations or {}
        proposals: list[tuple[str, str, str]] = []
        for item in ranked:
            seed = item.candidate.chunk_id
            proposals.append((seed, "reranked", seed))
            chunk = chunks.get(seed)
            if chunk and chunk.parent_chunk_id: proposals.append((chunk.parent_chunk_id, "parent", seed))
            for neighbor in adjacency.get(seed, (None, None)):
                if neighbor: proposals.append((neighbor, "adjacent", seed))
            for related in special_relations.get(seed, ()):
                proposals.append((related, "special_relation", seed))
        items: list[ContextItem] = []
        omitted: list[str] = []
        used = 0
        seen_ids: set[str] = set()
        seen_content: set[tuple[str, str]] = set()
        for chunk_id, reason, seed in proposals:
            if chunk_id in seen_ids: continue
            seen_ids.add(chunk_id)
            chunk = chunks.get(chunk_id)
            if chunk is None or (chunk.quality_score is not None and chunk.quality_score < self.min_quality):
                omitted.append(chunk_id); continue
            paper_id = paper_ids.get(chunk_id, "")
            content_key = (paper_id, re.sub(r"\s+", " ", chunk.text).strip().casefold())
            if content_key in seen_content: continue
            count = len(self.tokenizer.encode(chunk.text, add_special_tokens=False))
            if used + count > budget:
                omitted.append(chunk_id); continue
            seen_content.add(content_key)
            items.append(ContextItem(chunk, paper_id, reason, (seed,)))
            used += count
        return EvidenceContext(tuple(items), used, budget, tuple(dict.fromkeys(omitted)))
