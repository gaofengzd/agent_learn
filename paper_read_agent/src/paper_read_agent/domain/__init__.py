"""Framework-independent domain models."""

from paper_read_agent.domain.evidence import Citation, Evidence, EvidenceRegistry
from paper_read_agent.domain.models import (
    AnswerStatus,
    Chunk,
    ContentBlock,
    Conversation,
    Message,
    MessageRole,
    Page,
    Paper,
    PaperStatus,
    PaperVersion,
    QualityLevel,
    QualityReport,
)

__all__ = [
    "Citation", "Evidence", "EvidenceRegistry",
    "AnswerStatus",
    "Chunk",
    "ContentBlock",
    "Conversation",
    "Message",
    "MessageRole",
    "Page",
    "Paper",
    "PaperStatus",
    "PaperVersion",
    "QualityLevel",
    "QualityReport",
]
