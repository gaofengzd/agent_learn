from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

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
from paper_read_agent.persistence import SQLiteDatabase, SQLiteDomainRepository
from paper_read_agent.persistence.database import MIGRATIONS


@pytest.fixture
def database(tmp_path: Path) -> SQLiteDatabase:
    value = SQLiteDatabase(tmp_path / "paper_agent.sqlite3")
    value.initialize()
    return value


@pytest.fixture
def repository(database: SQLiteDatabase) -> SQLiteDomainRepository:
    return SQLiteDomainRepository(database)


def seed_paper(repository: SQLiteDomainRepository) -> tuple[Paper, PaperVersion]:
    paper = repository.create_paper(
        Paper(
            paper_id="paper-1",
            content_hash="paper-hash-1",
            title="A Paper",
            authors=("Author A", "Author B"),
            language="en",
            file_path="papers/paper-1.pdf",
            status=PaperStatus.PENDING,
        )
    )
    version = repository.create_version(
        PaperVersion(
            version_id="version-1",
            paper_id=paper.paper_id,
            content_hash=paper.content_hash,
            parser_name="docling",
        )
    )
    return paper, version


def test_initialize_is_idempotent_and_enables_foreign_keys(database: SQLiteDatabase) -> None:
    database.initialize()

    with database.connect() as connection:
        version = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        migration_count = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert version == MIGRATIONS[-1][0]
    assert migration_count == len(MIGRATIONS)
    assert foreign_keys == 1


def test_paper_create_query_and_update(repository: SQLiteDomainRepository) -> None:
    paper, _ = seed_paper(repository)

    loaded = repository.get_paper(paper.paper_id)
    assert loaded == paper

    updated = repository.update_paper(
        replace(
            loaded,
            title="Updated Paper",
            status=PaperStatus.READY,
            quality_level=QualityLevel.READY,
            page_count=12,
        )
    )
    reloaded = repository.get_paper(paper.paper_id)

    assert reloaded == updated
    assert reloaded.title == "Updated Paper"
    assert reloaded.status is PaperStatus.READY


def test_content_hash_is_unique(repository: SQLiteDomainRepository) -> None:
    seed_paper(repository)

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_paper(
            Paper(
                paper_id="paper-2",
                content_hash="paper-hash-1",
                title="Duplicate",
                file_path="papers/duplicate.pdf",
            )
        )


def test_foreign_key_rejects_orphan_version(repository: SQLiteDomainRepository) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_version(
            PaperVersion(
                version_id="orphan",
                paper_id="missing-paper",
                content_hash="orphan-hash",
            )
        )


def test_full_document_relationship_round_trip(repository: SQLiteDomainRepository) -> None:
    _, version = seed_paper(repository)
    page = repository.create_page(
        Page(
            page_id="page-1",
            version_id=version.version_id,
            pdf_page_number=1,
            native_text_coverage=0.95,
            quality_status=QualityLevel.READY,
        )
    )
    block = repository.create_block(
        ContentBlock(
            block_id="block-1",
            version_id=version.version_id,
            page_id=page.page_id,
            section_path=("Method",),
            block_type="paragraph",
            text="The proposed method uses retrieval.",
            bbox=(1.0, 2.0, 3.0, 4.0),
            quality_score=0.9,
        )
    )
    parent = repository.create_chunk(
        Chunk(
            chunk_id="parent-1",
            version_id=version.version_id,
            parent_chunk_id=None,
            text="Parent context",
            token_count=20,
            page_start=1,
            page_end=1,
            section_path=("Method",),
            content_type="parent",
            block_ids=(block.block_id,),
        )
    )
    child = repository.create_chunk(
        Chunk(
            chunk_id="child-1",
            version_id=version.version_id,
            parent_chunk_id=parent.chunk_id,
            text="retrieval",
            token_count=5,
            page_start=1,
            page_end=1,
            section_path=("Method",),
            content_type="paragraph",
            block_ids=(block.block_id,),
        )
    )

    assert repository.get_version(version.version_id) == version
    assert repository.get_page(page.page_id) == page
    assert repository.get_block(block.block_id) == block
    assert repository.get_chunk(parent.chunk_id) == parent
    assert repository.get_chunk(child.chunk_id) == child

    updated_version = repository.update_version(
        replace(version, parser_version="1.0", parse_status=PaperStatus.PARSING)
    )
    updated_page = repository.update_page(replace(page, printed_page_label="1", ocr_used=True))
    updated_block = repository.update_block(replace(block, text="Updated method text"))
    updated_child = repository.update_chunk(replace(child, text="updated retrieval", token_count=6))

    assert repository.get_version(version.version_id) == updated_version
    assert repository.get_page(page.page_id) == updated_page
    assert repository.get_block(block.block_id) == updated_block
    assert repository.get_chunk(child.chunk_id) == updated_child


def test_quality_report_create_query_and_update(repository: SQLiteDomainRepository) -> None:
    _, version = seed_paper(repository)
    report = repository.save_quality_report(
        QualityReport(
            quality_report_id="quality-1",
            version_id=version.version_id,
            page_coverage=0.8,
            missing_pages=(3,),
            warnings=("page missing",),
            final_level=QualityLevel.PARTIALLY_READY,
        )
    )
    updated = repository.save_quality_report(
        replace(
            report,
            page_coverage=1.0,
            missing_pages=(),
            warnings=(),
            final_level=QualityLevel.READY,
        )
    )

    assert repository.get_quality_report(report.quality_report_id) == updated


def test_conversation_and_message_round_trip(repository: SQLiteDomainRepository) -> None:
    paper, _ = seed_paper(repository)
    conversation = repository.create_conversation(
        Conversation(
            conversation_id="conversation-1",
            title="Paper discussion",
            scope_mode="selected",
            selected_paper_ids=(paper.paper_id,),
        )
    )
    message = repository.create_message(
        Message(
            message_id="message-1",
            conversation_id=conversation.conversation_id,
            role=MessageRole.ASSISTANT,
            content="Answer",
            structured_payload={"claims": []},
            retrieval_scope={"paper_ids": [paper.paper_id]},
            evidence_ids=("evidence-1",),
            answer_status=AnswerStatus.ANSWERED,
        )
    )

    assert repository.get_conversation(conversation.conversation_id) == conversation
    assert repository.get_message(message.message_id) == message

    updated_conversation = repository.update_conversation(
        replace(conversation, title="Updated discussion", scope_mode="library", selected_paper_ids=())
    )
    updated_message = repository.update_message(
        replace(message, content="Updated answer", answer_status=AnswerStatus.PARTIALLY_ANSWERED)
    )

    assert repository.get_conversation(conversation.conversation_id) == updated_conversation
    assert repository.get_message(message.message_id) == updated_message


def test_database_constraints_reject_invalid_enum_and_page_number(database: SQLiteDatabase) -> None:
    with database.transaction() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO papers
                (paper_id,content_hash,title,file_path,status,created_at,updated_at)
                VALUES ('bad','bad-hash','Bad','bad.pdf','unknown','now','now')"""
            )

    repository = SQLiteDomainRepository(database)
    _, version = seed_paper(repository)
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_page(
            Page(page_id="bad-page", version_id=version.version_id, pdf_page_number=0)
        )


def test_cascade_delete_prevents_orphan_document_records(database: SQLiteDatabase) -> None:
    repository = SQLiteDomainRepository(database)
    paper, version = seed_paper(repository)
    page = repository.create_page(Page(page_id="page-1", version_id=version.version_id, pdf_page_number=1))
    repository.create_block(
        ContentBlock(
            block_id="block-1",
            version_id=version.version_id,
            page_id=page.page_id,
            section_path=(),
            block_type="paragraph",
            text="text",
        )
    )

    with database.transaction() as connection:
        connection.execute("DELETE FROM papers WHERE paper_id=?", (paper.paper_id,))

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM paper_versions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0] == 0
