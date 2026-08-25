"""Core domain entities shared by application and persistence layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PaperStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    READY = "ready"
    PARTIALLY_READY = "partially_ready"
    FAILED = "failed"


class QualityLevel(StrEnum):
    READY = "ready"
    PARTIALLY_READY = "partially_ready"
    FAILED = "failed"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    CONFLICTED = "conflicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DOCUMENT_QUALITY_FAILURE = "document_quality_failure"
    OUT_OF_SCOPE = "out_of_scope"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Paper:
    paper_id: str
    content_hash: str
    title: str
    authors: tuple[str, ...] = ()
    language: str | None = None
    file_path: str = ""
    page_count: int | None = None
    status: PaperStatus = PaperStatus.PENDING
    quality_level: QualityLevel | None = None
    active_version_id: str | None = None
    created_at: str = ""
    updated_at: str = ""

@dataclass(frozen=True, slots=True)
class PaperVersion:
    version_id: str
    paper_id: str
    content_hash: str
    parser_name: str | None = None
    parser_version: str | None = None
    ocr_name: str | None = None
    ocr_version: str | None = None
    embedding_model_id: str | None = None
    parse_status: PaperStatus = PaperStatus.PENDING
    quality_report_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class Page:
    page_id: str
    version_id: str
    pdf_page_number: int
    printed_page_label: str | None = None
    native_text_coverage: float | None = None
    ocr_used: bool = False
    ocr_confidence: float | None = None
    quality_status: QualityLevel | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class ContentBlock:
    block_id: str
    version_id: str
    page_id: str
    section_path: tuple[str, ...]
    block_type: str
    text: str
    bbox: tuple[float, float, float, float] | None = None
    source_type: str = "native_pdf"
    quality_score: float | None = None
    previous_block_id: str | None = None
    next_block_id: str | None = None
    related_block_ids: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    version_id: str
    parent_chunk_id: str | None
    text: str
    token_count: int
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    content_type: str
    quality_score: float | None = None
    index_version: str | None = None
    block_ids: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class QualityReport:
    quality_report_id: str
    version_id: str
    page_coverage: float | None = None
    garbled_ratio: float | None = None
    ocr_confidence: float | None = None
    missing_pages: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    final_level: QualityLevel = QualityLevel.FAILED
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class Conversation:
    conversation_id: str
    title: str
    scope_mode: str
    selected_paper_ids: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class Message:
    message_id: str
    conversation_id: str
    role: MessageRole
    content: str
    structured_payload: dict[str, Any] = field(default_factory=dict)
    retrieval_scope: dict[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    answer_status: AnswerStatus | None = None
    created_at: str = ""
    updated_at: str = ""
