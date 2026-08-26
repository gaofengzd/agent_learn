"""Local cross-encoder reranking with explicit degradation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paper_read_agent.retrieval.hybrid import HybridCandidate
from paper_read_agent.retrieval.query_planner import QueryIntent, RetrievalPlan


class PairScorer(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class LocalBGEReranker:
    def __init__(self, model_path: str | Path, *, batch_size: int = 8) -> None:
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Local reranker model directory does not exist: {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        self.model.eval()
        self.batch_size = batch_size

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded = self.tokenizer(
                [[query, text] for text in batch], padding=True, truncation=True,
                max_length=8192, return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self.model(**encoded).logits.view(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    candidate: HybridCandidate
    score: float
    matched_subqueries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RerankResult:
    candidates: tuple[RerankedCandidate, ...]
    degraded: bool
    error: str | None = None


class CandidateReranker:
    def __init__(self, scorer: PairScorer, *, result_limit: int = 12,
                 min_score: float = 0.1) -> None:
        if not 8 <= result_limit <= 12:
            raise ValueError("Rerank result limit must be between 8 and 12")
        self.scorer = scorer
        self.result_limit = result_limit
        self.min_score = min_score

    def rerank(self, plan: RetrievalPlan, candidates: Sequence[HybridCandidate]) -> RerankResult:
        if not candidates:
            return RerankResult((), False)
        queries = tuple(dict.fromkeys((plan.original_question, *plan.subqueries)))
        texts = [item.text for item in candidates]
        try:
            per_query = [self.scorer.score(query, texts) for query in queries]
            if any(len(scores) != len(candidates) for scores in per_query):
                raise ValueError("Reranker returned an invalid score count")
        except Exception as exc:
            fallback = tuple(
                RerankedCandidate(item, item.rrf_score, ())
                for item in candidates[:self.result_limit]
            )
            return RerankResult(fallback, True, f"{type(exc).__name__}: {exc}")
        values = []
        for index, candidate in enumerate(candidates):
            scores = [row[index] for row in per_query]
            best = max(scores)
            if best < self.min_score:
                continue
            matched = tuple(query for query, score in zip(queries, scores, strict=True)
                            if score >= self.min_score)
            values.append(RerankedCandidate(candidate, best, matched))
        values.sort(key=lambda item: (-item.score, -item.candidate.rrf_score,
                                      item.candidate.chunk_id))
        if plan.intent is QueryIntent.COMPARISON:
            selected: list[RerankedCandidate] = []
            for paper_id in plan.paper_ids:
                match = next((item for item in values
                              if item.candidate.metadata.get("paper_id") == paper_id), None)
                if match is not None and match not in selected:
                    selected.append(match)
            selected.extend(item for item in values if item not in selected)
            values = selected
        return RerankResult(tuple(values[:self.result_limit]), False)
