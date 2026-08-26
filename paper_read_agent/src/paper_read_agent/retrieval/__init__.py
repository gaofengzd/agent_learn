"""Retrieval pipeline boundary."""
"""Retrieval and indexing components."""

from paper_read_agent.retrieval.vector_index import (
    ChromaVectorIndex,
    LocalBGEEmbedder,
    VectorHit,
    VectorIndexError,
    VectorRecord,
)
from paper_read_agent.retrieval.keyword_index import KeywordHit, SQLiteKeywordIndex
from paper_read_agent.retrieval.query_planner import QueryIntent, QueryPlanner, RetrievalPlan
from paper_read_agent.retrieval.hybrid import (
    HybridCandidate, HybridRecallResult, HybridRetriever, RecallTrace,
)
from paper_read_agent.retrieval.reranker import (
    CandidateReranker, LocalBGEReranker, RerankedCandidate, RerankResult,
)
from paper_read_agent.retrieval.context_builder import ContextBuilder, ContextItem, EvidenceContext

__all__ = [
    "ChromaVectorIndex", "LocalBGEEmbedder", "VectorHit", "VectorIndexError", "VectorRecord",
    "KeywordHit", "SQLiteKeywordIndex",
    "QueryIntent", "QueryPlanner", "RetrievalPlan",
    "HybridCandidate", "HybridRecallResult", "HybridRetriever", "RecallTrace",
    "CandidateReranker", "LocalBGEReranker", "RerankedCandidate", "RerankResult",
    "ContextBuilder", "ContextItem", "EvidenceContext",
]
