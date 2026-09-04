"""Repository contracts and SQLite implementation for core entities."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
import sqlite3
from typing import Protocol

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
from paper_read_agent.persistence.database import SQLiteDatabase


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class DomainRepository(Protocol):
    def create_paper(self, paper: Paper) -> Paper: ...
    def get_paper(self, paper_id: str) -> Paper | None: ...
    def get_paper_by_content_hash(self, content_hash: str) -> Paper | None: ...
    def update_paper(self, paper: Paper) -> Paper: ...
    def create_version(self, version: PaperVersion) -> PaperVersion: ...
    def get_version(self, version_id: str) -> PaperVersion | None: ...
    def update_version(self, version: PaperVersion) -> PaperVersion: ...
    def create_page(self, page: Page) -> Page: ...
    def get_page(self, page_id: str) -> Page | None: ...
    def update_page(self, page: Page) -> Page: ...
    def create_block(self, block: ContentBlock) -> ContentBlock: ...
    def get_block(self, block_id: str) -> ContentBlock | None: ...
    def update_block(self, block: ContentBlock) -> ContentBlock: ...
    def create_chunk(self, chunk: Chunk) -> Chunk: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...
    def list_chunks(self, version_id: str) -> tuple[Chunk, ...]: ...
    def update_chunk(self, chunk: Chunk) -> Chunk: ...
    def save_quality_report(self, report: QualityReport) -> QualityReport: ...
    def get_quality_report(self, report_id: str) -> QualityReport | None: ...
    def create_conversation(self, conversation: Conversation) -> Conversation: ...
    def get_conversation(self, conversation_id: str) -> Conversation | None: ...
    def update_conversation(self, conversation: Conversation) -> Conversation: ...
    def create_message(self, message: Message) -> Message: ...
    def get_message(self, message_id: str) -> Message | None: ...
    def update_message(self, message: Message) -> Message: ...


class SQLiteDomainRepository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def create_paper(self, paper: Paper) -> Paper:
        now = utc_now()
        value = replace(paper, created_at=paper.created_at or now, updated_at=paper.updated_at or now)
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    value.paper_id, value.content_hash, value.title, _json(value.authors),
                    value.language, value.file_path, value.page_count, value.status.value,
                    value.quality_level.value if value.quality_level else None,
                    value.active_version_id, value.created_at, value.updated_at,
                ),
            )
        return value

    def get_paper(self, paper_id: str) -> Paper | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        return self._paper(row) if row else None

    def get_paper_by_content_hash(self, content_hash: str) -> Paper | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE content_hash=?", (content_hash,)
            ).fetchone()
        return self._paper(row) if row else None

    def list_papers(self) -> tuple[Paper, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM papers ORDER BY created_at DESC, paper_id"
            ).fetchall()
        return tuple(self._paper(row) for row in rows)

    def delete_paper_record(self, paper_id: str) -> None:
        """Rollback helper for a paper whose initial file persistence failed."""
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))

    def update_paper(self, paper: Paper) -> Paper:
        value = replace(paper, updated_at=utc_now())
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """UPDATE papers SET content_hash=?,title=?,authors_json=?,language=?,file_path=?,
                page_count=?,status=?,quality_level=?,active_version_id=?,updated_at=? WHERE paper_id=?""",
                (
                    value.content_hash, value.title, _json(value.authors), value.language,
                    value.file_path, value.page_count, value.status.value,
                    value.quality_level.value if value.quality_level else None,
                    value.active_version_id, value.updated_at, value.paper_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(value.paper_id)
        return value

    def create_version(self, version: PaperVersion) -> PaperVersion:
        now = utc_now()
        value = replace(version, created_at=version.created_at or now, updated_at=version.updated_at or now)
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO paper_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    value.version_id, value.paper_id, value.content_hash, value.parser_name,
                    value.parser_version, value.ocr_name, value.ocr_version,
                    value.embedding_model_id, value.parse_status.value, value.quality_report_id,
                    value.created_at, value.updated_at,
                ),
            )
        return value

    def get_version(self, version_id: str) -> PaperVersion | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM paper_versions WHERE version_id=?", (version_id,)).fetchone()
        return self._version(row) if row else None

    def update_version(self, version: PaperVersion) -> PaperVersion:
        value = replace(version, updated_at=utc_now())
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """UPDATE paper_versions SET paper_id=?,content_hash=?,parser_name=?,parser_version=?,
                ocr_name=?,ocr_version=?,embedding_model_id=?,parse_status=?,quality_report_id=?,
                updated_at=? WHERE version_id=?""",
                (
                    value.paper_id, value.content_hash, value.parser_name, value.parser_version,
                    value.ocr_name, value.ocr_version, value.embedding_model_id,
                    value.parse_status.value, value.quality_report_id, value.updated_at,
                    value.version_id,
                ),
            )
            self._require_updated(cursor, value.version_id)
        return value

    def create_page(self, page: Page) -> Page:
        now = utc_now()
        value = replace(page, created_at=page.created_at or now, updated_at=page.updated_at or now)
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    value.page_id, value.version_id, value.pdf_page_number,
                    value.printed_page_label, value.native_text_coverage, int(value.ocr_used),
                    value.ocr_confidence, value.quality_status.value if value.quality_status else None,
                    value.created_at, value.updated_at,
                ),
            )
        return value

    def get_page(self, page_id: str) -> Page | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM pages WHERE page_id=?", (page_id,)).fetchone()
        return self._page(row) if row else None

    def update_page(self, page: Page) -> Page:
        value = replace(page, updated_at=utc_now())
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """UPDATE pages SET version_id=?,pdf_page_number=?,printed_page_label=?,
                native_text_coverage=?,ocr_used=?,ocr_confidence=?,quality_status=?,updated_at=?
                WHERE page_id=?""",
                (
                    value.version_id, value.pdf_page_number, value.printed_page_label,
                    value.native_text_coverage, int(value.ocr_used), value.ocr_confidence,
                    value.quality_status.value if value.quality_status else None,
                    value.updated_at, value.page_id,
                ),
            )
            self._require_updated(cursor, value.page_id)
        return value

    def create_block(self, block: ContentBlock) -> ContentBlock:
        now = utc_now()
        value = replace(block, created_at=block.created_at or now, updated_at=block.updated_at or now)
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO content_blocks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    value.block_id, value.version_id, value.page_id, _json(value.section_path),
                    value.block_type, value.text, _json(value.bbox) if value.bbox else None,
                    value.source_type, value.quality_score, value.previous_block_id,
                    value.next_block_id, _json(value.related_block_ids), value.created_at,
                    value.updated_at,
                ),
            )
        return value

    def get_block(self, block_id: str) -> ContentBlock | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM content_blocks WHERE block_id=?", (block_id,)).fetchone()
        return self._block(row) if row else None

    def update_block(self, block: ContentBlock) -> ContentBlock:
        value = replace(block, updated_at=utc_now())
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """UPDATE content_blocks SET version_id=?,page_id=?,section_path_json=?,block_type=?,
                text=?,bbox_json=?,source_type=?,quality_score=?,previous_block_id=?,next_block_id=?,
                related_block_ids_json=?,updated_at=? WHERE block_id=?""",
                (
                    value.version_id, value.page_id, _json(value.section_path), value.block_type,
                    value.text, _json(value.bbox) if value.bbox else None, value.source_type,
                    value.quality_score, value.previous_block_id, value.next_block_id,
                    _json(value.related_block_ids), value.updated_at, value.block_id,
                ),
            )
            self._require_updated(cursor, value.block_id)
        return value

    def create_chunk(self, chunk: Chunk) -> Chunk:
        now = utc_now()
        value = replace(chunk, created_at=chunk.created_at or now, updated_at=chunk.updated_at or now)
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    value.chunk_id, value.version_id, value.parent_chunk_id, value.text,
                    value.token_count, value.page_start, value.page_end,
                    _json(value.section_path), value.content_type, value.quality_score,
                    value.index_version, value.created_at, value.updated_at,
                ),
            )
            for position, block_id in enumerate(value.block_ids):
                connection.execute(
                    "INSERT INTO chunk_blocks(chunk_id,block_id,position) VALUES (?,?,?)",
                    (value.chunk_id, block_id, position),
                )
        return value

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
            if not row:
                return None
            block_ids = tuple(
                item["block_id"]
                for item in connection.execute(
                    "SELECT block_id FROM chunk_blocks WHERE chunk_id=? ORDER BY position", (chunk_id,)
                )
            )
        return self._chunk(row, block_ids)

    def list_chunks(self, version_id: str) -> tuple[Chunk, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE version_id=? "
                "ORDER BY page_start,page_end,chunk_id", (version_id,)
            ).fetchall()
            values = []
            for row in rows:
                block_ids = tuple(
                    item["block_id"]
                    for item in connection.execute(
                        "SELECT block_id FROM chunk_blocks WHERE chunk_id=? ORDER BY position",
                        (row["chunk_id"],),
                    )
                )
                values.append(self._chunk(row, block_ids))
        return tuple(values)

    def update_chunk(self, chunk: Chunk) -> Chunk:
        value = replace(chunk, updated_at=utc_now())
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """UPDATE chunks SET version_id=?,parent_chunk_id=?,text=?,token_count=?,
                page_start=?,page_end=?,section_path_json=?,content_type=?,quality_score=?,
                index_version=?,updated_at=? WHERE chunk_id=?""",
                (
                    value.version_id, value.parent_chunk_id, value.text, value.token_count,
                    value.page_start, value.page_end, _json(value.section_path), value.content_type,
                    value.quality_score, value.index_version, value.updated_at, value.chunk_id,
                ),
            )
            self._require_updated(cursor, value.chunk_id)
            connection.execute("DELETE FROM chunk_blocks WHERE chunk_id=?", (value.chunk_id,))
            for position, block_id in enumerate(value.block_ids):
                connection.execute(
                    "INSERT INTO chunk_blocks(chunk_id,block_id,position) VALUES (?,?,?)",
                    (value.chunk_id, block_id, position),
                )
        return value

    def save_quality_report(self, report: QualityReport) -> QualityReport:
        now = utc_now()
        value = replace(report, created_at=report.created_at or now, updated_at=now)
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO quality_reports VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(quality_report_id) DO UPDATE SET
                page_coverage=excluded.page_coverage,garbled_ratio=excluded.garbled_ratio,
                ocr_confidence=excluded.ocr_confidence,missing_pages_json=excluded.missing_pages_json,
                warnings_json=excluded.warnings_json,final_level=excluded.final_level,
                updated_at=excluded.updated_at""",
                (
                    value.quality_report_id, value.version_id, value.page_coverage,
                    value.garbled_ratio, value.ocr_confidence, _json(value.missing_pages),
                    _json(value.warnings), value.final_level.value, value.created_at, value.updated_at,
                ),
            )
        return value

    def get_quality_report(self, report_id: str) -> QualityReport | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM quality_reports WHERE quality_report_id=?", (report_id,)
            ).fetchone()
        return self._quality_report(row) if row else None

    def create_conversation(self, conversation: Conversation) -> Conversation:
        now = utc_now()
        value = replace(
            conversation,
            created_at=conversation.created_at or now,
            updated_at=conversation.updated_at or now,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO conversations VALUES (?,?,?,?,?)",
                (value.conversation_id, value.title, value.scope_mode, value.created_at, value.updated_at),
            )
            for paper_id in value.selected_paper_ids:
                connection.execute(
                    "INSERT INTO conversation_papers(conversation_id,paper_id) VALUES (?,?)",
                    (value.conversation_id, paper_id),
                )
        return value

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            if not row:
                return None
            paper_ids = tuple(
                item["paper_id"]
                for item in connection.execute(
                    "SELECT paper_id FROM conversation_papers WHERE conversation_id=? ORDER BY paper_id",
                    (conversation_id,),
                )
            )
        return Conversation(
            conversation_id=row["conversation_id"], title=row["title"],
            scope_mode=row["scope_mode"], selected_paper_ids=paper_ids,
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def update_conversation(self, conversation: Conversation) -> Conversation:
        value = replace(conversation, updated_at=utc_now())
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title=?,scope_mode=?,updated_at=? WHERE conversation_id=?",
                (value.title, value.scope_mode, value.updated_at, value.conversation_id),
            )
            self._require_updated(cursor, value.conversation_id)
            connection.execute(
                "DELETE FROM conversation_papers WHERE conversation_id=?", (value.conversation_id,)
            )
            for paper_id in value.selected_paper_ids:
                connection.execute(
                    "INSERT INTO conversation_papers(conversation_id,paper_id) VALUES (?,?)",
                    (value.conversation_id, paper_id),
                )
        return value

    def list_conversations(self) -> tuple[Conversation, ...]:
        with self.database.connect() as connection:
            ids = [row[0] for row in connection.execute("SELECT conversation_id FROM conversations ORDER BY updated_at DESC")]
        return tuple(item for item in (self.get_conversation(cid) for cid in ids) if item is not None)

    def delete_conversation(self, conversation_id: str) -> None:
        with self.database.transaction() as connection:
            cursor = connection.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))
            self._require_updated(cursor, conversation_id)

    def list_messages(self, conversation_id: str) -> tuple[Message, ...]:
        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at,message_id",
                                      (conversation_id,)).fetchall()
        return tuple(self._message(row) for row in rows)

    def create_message(self, message: Message) -> Message:
        now = utc_now()
        value = replace(message, created_at=message.created_at or now, updated_at=message.updated_at or now)
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    value.message_id, value.conversation_id, value.role.value, value.content,
                    _json(value.structured_payload), _json(value.retrieval_scope),
                    _json(value.evidence_ids), value.answer_status.value if value.answer_status else None,
                    value.created_at, value.updated_at,
                ),
            )
        return value

    def create_message_pair(self, user: Message, assistant: Message) -> tuple[Message, Message]:
        now = utc_now()
        values = tuple(replace(item, created_at=item.created_at or now,
                               updated_at=item.updated_at or now) for item in (user, assistant))
        with self.database.transaction() as connection:
            for value in values:
                connection.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    value.message_id, value.conversation_id, value.role.value, value.content,
                    _json(value.structured_payload), _json(value.retrieval_scope),
                    _json(value.evidence_ids), value.answer_status.value if value.answer_status else None,
                    value.created_at, value.updated_at))
        return values  # type: ignore[return-value]

    def get_message(self, message_id: str) -> Message | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM messages WHERE message_id=?", (message_id,)).fetchone()
        return self._message(row) if row else None

    def update_message(self, message: Message) -> Message:
        value = replace(message, updated_at=utc_now())
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """UPDATE messages SET conversation_id=?,role=?,content=?,structured_payload_json=?,
                retrieval_scope_json=?,evidence_ids_json=?,answer_status=?,updated_at=? WHERE message_id=?""",
                (
                    value.conversation_id, value.role.value, value.content,
                    _json(value.structured_payload), _json(value.retrieval_scope),
                    _json(value.evidence_ids), value.answer_status.value if value.answer_status else None,
                    value.updated_at, value.message_id,
                ),
            )
            self._require_updated(cursor, value.message_id)
        return value

    @staticmethod
    def _require_updated(cursor: sqlite3.Cursor, entity_id: str) -> None:
        if cursor.rowcount != 1:
            raise KeyError(entity_id)

    @staticmethod
    def _paper(row: sqlite3.Row) -> Paper:
        return Paper(
            paper_id=row["paper_id"], content_hash=row["content_hash"], title=row["title"],
            authors=tuple(json.loads(row["authors_json"])), language=row["language"],
            file_path=row["file_path"], page_count=row["page_count"], status=PaperStatus(row["status"]),
            quality_level=QualityLevel(row["quality_level"]) if row["quality_level"] else None,
            active_version_id=row["active_version_id"], created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _version(row: sqlite3.Row) -> PaperVersion:
        return PaperVersion(
            version_id=row["version_id"], paper_id=row["paper_id"], content_hash=row["content_hash"],
            parser_name=row["parser_name"], parser_version=row["parser_version"], ocr_name=row["ocr_name"],
            ocr_version=row["ocr_version"], embedding_model_id=row["embedding_model_id"],
            parse_status=PaperStatus(row["parse_status"]), quality_report_id=row["quality_report_id"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _page(row: sqlite3.Row) -> Page:
        return Page(
            page_id=row["page_id"], version_id=row["version_id"], pdf_page_number=row["pdf_page_number"],
            printed_page_label=row["printed_page_label"], native_text_coverage=row["native_text_coverage"],
            ocr_used=bool(row["ocr_used"]), ocr_confidence=row["ocr_confidence"],
            quality_status=QualityLevel(row["quality_status"]) if row["quality_status"] else None,
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _block(row: sqlite3.Row) -> ContentBlock:
        return ContentBlock(
            block_id=row["block_id"], version_id=row["version_id"], page_id=row["page_id"],
            section_path=tuple(json.loads(row["section_path_json"])), block_type=row["block_type"],
            text=row["text"], bbox=tuple(json.loads(row["bbox_json"])) if row["bbox_json"] else None,
            source_type=row["source_type"], quality_score=row["quality_score"],
            previous_block_id=row["previous_block_id"], next_block_id=row["next_block_id"],
            related_block_ids=tuple(json.loads(row["related_block_ids_json"])),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _chunk(row: sqlite3.Row, block_ids: tuple[str, ...]) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"], version_id=row["version_id"], parent_chunk_id=row["parent_chunk_id"],
            text=row["text"], token_count=row["token_count"], page_start=row["page_start"],
            page_end=row["page_end"], section_path=tuple(json.loads(row["section_path_json"])),
            content_type=row["content_type"], quality_score=row["quality_score"],
            index_version=row["index_version"], block_ids=block_ids,
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _quality_report(row: sqlite3.Row) -> QualityReport:
        return QualityReport(
            quality_report_id=row["quality_report_id"], version_id=row["version_id"],
            page_coverage=row["page_coverage"], garbled_ratio=row["garbled_ratio"],
            ocr_confidence=row["ocr_confidence"], missing_pages=tuple(json.loads(row["missing_pages_json"])),
            warnings=tuple(json.loads(row["warnings_json"])), final_level=QualityLevel(row["final_level"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _message(row: sqlite3.Row) -> Message:
        return Message(
            message_id=row["message_id"], conversation_id=row["conversation_id"],
            role=MessageRole(row["role"]), content=row["content"],
            structured_payload=json.loads(row["structured_payload_json"]),
            retrieval_scope=json.loads(row["retrieval_scope_json"]),
            evidence_ids=tuple(json.loads(row["evidence_ids_json"])),
            answer_status=AnswerStatus(row["answer_status"]) if row["answer_status"] else None,
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
