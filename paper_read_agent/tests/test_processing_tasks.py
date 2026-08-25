from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

import pytest

from paper_read_agent.application.processing_tasks import (
    ProcessingOutcome,
    ProcessingTaskRunner,
    ProcessingTaskStore,
    TaskStatus,
)
from paper_read_agent.domain.models import Paper, PaperStatus, PaperVersion, QualityLevel
from paper_read_agent.persistence import SQLiteDatabase, SQLiteDomainRepository


@pytest.fixture
def setup(tmp_path: Path):
    database = SQLiteDatabase(tmp_path / "db.sqlite3")
    database.initialize()
    repository = SQLiteDomainRepository(database)
    paper = repository.create_paper(
        Paper(paper_id="paper", content_hash="hash", title="Paper", file_path="paper.pdf")
    )
    version = repository.create_version(
        PaperVersion(version_id="version", paper_id=paper.paper_id, content_hash="hash")
    )
    return database, repository, version


def test_async_task_reaches_ready_state(setup) -> None:
    database, repository, version = setup
    runner = ProcessingTaskRunner(
        repository,
        ProcessingTaskStore(database),
        lambda _version_id: ProcessingOutcome(PaperStatus.READY, QualityLevel.READY),
    )
    try:
        queued = runner.enqueue(version.version_id)
        finished = runner.future(queued.task_id).result(timeout=3)
    finally:
        runner.shutdown()

    assert queued.status is TaskStatus.QUEUED
    assert finished.status is TaskStatus.SUCCEEDED
    assert repository.get_version(version.version_id).parse_status is PaperStatus.READY
    assert repository.get_paper("paper").status is PaperStatus.READY


def test_processing_failure_records_stage_and_marks_document_failed(setup) -> None:
    database, repository, version = setup

    def fail(_version_id):
        raise ValueError("parser broke")

    runner = ProcessingTaskRunner(repository, ProcessingTaskStore(database), fail)
    try:
        task = runner.enqueue(version.version_id)
        finished = runner.future(task.task_id).result(timeout=3)
    finally:
        runner.shutdown()

    assert finished.status is TaskStatus.FAILED
    assert finished.error_stage == "processing"
    assert "ValueError: parser broke" in finished.error_message
    assert repository.get_version(version.version_id).parse_status is PaperStatus.FAILED
    assert repository.get_paper("paper").status is PaperStatus.FAILED


def test_duplicate_active_task_is_rejected(setup) -> None:
    database, _repository, version = setup
    store = ProcessingTaskStore(database)
    first = store.create(version.version_id)

    with pytest.raises(sqlite3.IntegrityError):
        store.create(version.version_id)

    assert store.get(first.task_id) == first


def test_only_failed_task_can_be_retried_and_attempt_increments(setup) -> None:
    database, repository, version = setup
    store = ProcessingTaskStore(database)
    first = store.create(version.version_id)
    store.mark_failed(first.task_id, stage="test", message="failed")
    runner = ProcessingTaskRunner(
        repository, store,
        lambda _version_id: ProcessingOutcome(PaperStatus.PARTIALLY_READY, QualityLevel.PARTIALLY_READY),
    )
    try:
        retry = runner.retry(version.version_id)
        finished = runner.future(retry.task_id).result(timeout=3)
    finally:
        runner.shutdown()

    assert retry.attempt == 2
    assert finished.status is TaskStatus.SUCCEEDED
    assert repository.get_paper("paper").status is PaperStatus.PARTIALLY_READY


def test_retry_rejects_non_failed_latest_task(setup) -> None:
    database, repository, version = setup
    store = ProcessingTaskStore(database)
    store.create(version.version_id)
    runner = ProcessingTaskRunner(repository, store, lambda _: ProcessingOutcome(PaperStatus.READY))
    try:
        with pytest.raises(RuntimeError, match="Only a failed task"):
            runner.retry(version.version_id)
    finally:
        runner.shutdown()


def test_recover_stale_running_task(setup) -> None:
    database, repository, version = setup
    store = ProcessingTaskStore(database)
    task = store.create(version.version_id)
    store.mark_running(task.task_id)
    old = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    with database.transaction() as connection:
        connection.execute(
            "UPDATE processing_tasks SET updated_at=? WHERE task_id=?", (old, task.task_id)
        )
    repository.update_version(replace(version, parse_status=PaperStatus.PARSING))
    runner = ProcessingTaskRunner(repository, store, lambda _: ProcessingOutcome(PaperStatus.READY))
    try:
        recovered = runner.recover_stale(older_than=timedelta(hours=1))
    finally:
        runner.shutdown()

    assert recovered == (task.task_id,)
    assert store.get(task.task_id).status is TaskStatus.FAILED
    assert store.get(task.task_id).error_stage == "recovery"
    assert repository.get_version(version.version_id).parse_status is PaperStatus.FAILED
