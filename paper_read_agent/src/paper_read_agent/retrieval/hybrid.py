"""Hybrid keyword/vector recall with reciprocal-rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol, Sequence

from paper_read_agent.retrieval.query_planner import RetrievalPlan


class Retriever(Protocol):
    def query(self, text: str, **kwargs: object) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class RecallTrace:
    route: str
    query: str
    rank: int


@dataclass(frozen=True, slots=True)
class HybridCandidate:
    chunk_id: str
    text: str
    rrf_score: float
    metadata: dict[str, object]
    traces: tuple[RecallTrace, ...]


@dataclass(frozen=True, slots=True)
class HybridRecallResult:
    candidates: tuple[HybridCandidate, ...]
    route_failures: tuple[str, ...]
    degraded: bool


class HybridRetriever:
    def __init__(
        self,
        vector: Retriever,
        keyword: Retriever,
        *,
        candidate_limit: int = 50,
        per_route_limit: int = 50,
        rrf_k: int = 60,
    ) -> None:
        if not 30 <= candidate_limit <= 50:
            raise ValueError("Hybrid candidate limit must be between 30 and 50")
        if per_route_limit <= 0 or rrf_k <= 0:
            raise ValueError("Per-route limit and RRF k must be positive")
        self.vector = vector
        self.keyword = keyword
        self.candidate_limit = candidate_limit
        self.per_route_limit = per_route_limit
        self.rrf_k = rrf_k

    def retrieve(self, plan: RetrievalPlan) -> HybridRecallResult:
        if plan.needs_clarification or not plan.paper_ids:
            return HybridRecallResult((), (), False)
        queries = tuple(dict.fromkeys((plan.original_question, *plan.subqueries)))
        scores: dict[str, float] = {}
        values: dict[str, tuple[str, dict[str, object]]] = {}
        traces: dict[str, list[RecallTrace]] = {}
        failures: list[str] = []
        route_success = {"vector": False, "keyword": False}
        for query in queries:
            for route, retriever in (("vector", self.vector), ("keyword", self.keyword)):
                try:
                    hits = retriever.query(
                        query,
                        limit=self.per_route_limit,
                        paper_ids=list(plan.paper_ids),
                        include_references=plan.include_references,
                    )
                    route_success[route] = True
                except Exception as exc:
                    failures.append(f"{route} query failed for {query!r}: {type(exc).__name__}: {exc}")
                    continue
                for rank, hit in enumerate(hits, start=1):
                    chunk_id, text, metadata = self._hit(hit)
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
                    values.setdefault(chunk_id, (text, metadata))
                    traces.setdefault(chunk_id, []).append(RecallTrace(route, query, rank))

        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        candidates: list[HybridCandidate] = []
        seen_content: set[tuple[str, str]] = set()
        for chunk_id in ranked:
            text, metadata = values[chunk_id]
            content_key = (
                str(metadata.get("paper_id", "")),
                re.sub(r"\s+", " ", text).strip().casefold(),
            )
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            candidates.append(
                HybridCandidate(chunk_id, text, scores[chunk_id], metadata,
                                tuple(traces[chunk_id]))
            )
            if len(candidates) >= self.candidate_limit:
                break
        degraded = bool(failures) or not all(route_success.values())
        return HybridRecallResult(tuple(candidates), tuple(failures), degraded)

    @staticmethod
    def _hit(hit: Any) -> tuple[str, str, dict[str, object]]:
        metadata = dict(getattr(hit, "metadata", {}) or {})
        for name in ("paper_id", "version_id", "content_type"):
            value = getattr(hit, name, None)
            if value is not None:
                metadata.setdefault(name, value)
        return str(hit.chunk_id), str(hit.text), metadata
