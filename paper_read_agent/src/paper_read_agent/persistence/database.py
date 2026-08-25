"""SQLite connection management and schema migrations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3


MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL DEFAULT '[]',
            language TEXT,
            file_path TEXT NOT NULL,
            page_count INTEGER CHECK (page_count IS NULL OR page_count >= 0),
            status TEXT NOT NULL CHECK (status IN ('pending','parsing','ready','partially_ready','failed')),
            quality_level TEXT CHECK (quality_level IS NULL OR quality_level IN ('ready','partially_ready','failed')),
            active_version_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE paper_versions (
            version_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
            content_hash TEXT NOT NULL UNIQUE,
            parser_name TEXT,
            parser_version TEXT,
            ocr_name TEXT,
            ocr_version TEXT,
            embedding_model_id TEXT,
            parse_status TEXT NOT NULL CHECK (parse_status IN ('pending','parsing','ready','partially_ready','failed')),
            quality_report_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_paper_versions_paper_id ON paper_versions(paper_id);

        CREATE TABLE pages (
            page_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL REFERENCES paper_versions(version_id) ON DELETE CASCADE,
            pdf_page_number INTEGER NOT NULL CHECK (pdf_page_number > 0),
            printed_page_label TEXT,
            native_text_coverage REAL CHECK (native_text_coverage IS NULL OR native_text_coverage BETWEEN 0 AND 1),
            ocr_used INTEGER NOT NULL DEFAULT 0 CHECK (ocr_used IN (0,1)),
            ocr_confidence REAL CHECK (ocr_confidence IS NULL OR ocr_confidence BETWEEN 0 AND 1),
            quality_status TEXT CHECK (quality_status IS NULL OR quality_status IN ('ready','partially_ready','failed')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(version_id, pdf_page_number)
        );
        CREATE INDEX idx_pages_version_id ON pages(version_id);

        CREATE TABLE content_blocks (
            block_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL REFERENCES paper_versions(version_id) ON DELETE CASCADE,
            page_id TEXT NOT NULL REFERENCES pages(page_id) ON DELETE CASCADE,
            section_path_json TEXT NOT NULL DEFAULT '[]',
            block_type TEXT NOT NULL,
            text TEXT NOT NULL,
            bbox_json TEXT,
            source_type TEXT NOT NULL,
            quality_score REAL CHECK (quality_score IS NULL OR quality_score BETWEEN 0 AND 1),
            previous_block_id TEXT REFERENCES content_blocks(block_id) ON DELETE SET NULL,
            next_block_id TEXT REFERENCES content_blocks(block_id) ON DELETE SET NULL,
            related_block_ids_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_content_blocks_version_id ON content_blocks(version_id);
        CREATE INDEX idx_content_blocks_page_id ON content_blocks(page_id);

        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL REFERENCES paper_versions(version_id) ON DELETE CASCADE,
            parent_chunk_id TEXT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL CHECK (token_count > 0),
            page_start INTEGER NOT NULL CHECK (page_start > 0),
            page_end INTEGER NOT NULL CHECK (page_end >= page_start),
            section_path_json TEXT NOT NULL DEFAULT '[]',
            content_type TEXT NOT NULL,
            quality_score REAL CHECK (quality_score IS NULL OR quality_score BETWEEN 0 AND 1),
            index_version TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_chunks_version_id ON chunks(version_id);
        CREATE INDEX idx_chunks_parent_chunk_id ON chunks(parent_chunk_id);

        CREATE TABLE chunk_blocks (
            chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            block_id TEXT NOT NULL REFERENCES content_blocks(block_id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK (position >= 0),
            PRIMARY KEY (chunk_id, block_id),
            UNIQUE (chunk_id, position)
        );

        CREATE TABLE quality_reports (
            quality_report_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL UNIQUE REFERENCES paper_versions(version_id) ON DELETE CASCADE,
            page_coverage REAL CHECK (page_coverage IS NULL OR page_coverage BETWEEN 0 AND 1),
            garbled_ratio REAL CHECK (garbled_ratio IS NULL OR garbled_ratio BETWEEN 0 AND 1),
            ocr_confidence REAL CHECK (ocr_confidence IS NULL OR ocr_confidence BETWEEN 0 AND 1),
            missing_pages_json TEXT NOT NULL DEFAULT '[]',
            warnings_json TEXT NOT NULL DEFAULT '[]',
            final_level TEXT NOT NULL CHECK (final_level IN ('ready','partially_ready','failed')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            scope_mode TEXT NOT NULL CHECK (scope_mode IN ('selected','library')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE conversation_papers (
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
            PRIMARY KEY (conversation_id, paper_id)
        );

        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user','assistant')),
            content TEXT NOT NULL,
            structured_payload_json TEXT NOT NULL DEFAULT '{}',
            retrieval_scope_json TEXT NOT NULL DEFAULT '{}',
            evidence_ids_json TEXT NOT NULL DEFAULT '[]',
            answer_status TEXT CHECK (answer_status IS NULL OR answer_status IN (
                'answered','partially_answered','conflicted','insufficient_evidence',
                'document_quality_failure','out_of_scope'
            )),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_messages_conversation_id ON messages(conversation_id, created_at);
        """,
    ),
    (
        2,
        """
        CREATE TABLE processing_tasks (
            task_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL REFERENCES paper_versions(version_id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
            attempt INTEGER NOT NULL CHECK (attempt > 0),
            error_stage TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_processing_tasks_version_id ON processing_tasks(version_id, created_at);
        CREATE UNIQUE INDEX idx_processing_tasks_one_active_version
        ON processing_tasks(version_id)
        WHERE status IN ('queued','running');
        """,
    ),
)


class SQLiteDatabase:
    """Own SQLite connections and apply numbered, repeatable migrations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            applied = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for version, sql in MIGRATIONS:
                if version in applied:
                    continue
                migration_script = (
                    "BEGIN IMMEDIATE;\n"
                    + sql
                    + f"\nINSERT INTO schema_migrations(version) VALUES ({version});\n"
                    + "COMMIT;"
                )
                try:
                    connection.executescript(migration_script)
                except Exception:
                    if connection.in_transaction:
                        connection.rollback()
                    raise

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()
