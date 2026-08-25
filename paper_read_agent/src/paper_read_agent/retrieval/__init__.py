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

__all__ = [
    "ChromaVectorIndex", "LocalBGEEmbedder", "VectorHit", "VectorIndexError", "VectorRecord",
    "KeywordHit", "SQLiteKeywordIndex",
    "QueryIntent", "QueryPlanner", "RetrievalPlan",
    "HybridCandidate", "HybridRecallResult", "HybridRetriever", "RecallTrace",
]
