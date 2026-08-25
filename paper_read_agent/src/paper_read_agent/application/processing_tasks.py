"""Persistent background processing tasks and paper status transitions."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Lock
from uuid import uuid4
import sqlite3

from paper_read_agent.domain.models import PaperStatus, QualityLevel
from paper_read_agent.persistence.database import SQLiteDatabase
from paper_read_agent.persistence.repositories import SQLiteDomainRepository, utc_now


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProcessingTask:
    task_id: str
    version_id: str
    status: TaskStatus
    attempt: int
    error_stage: str | None = None
    error_message: str | None = None
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class ProcessingOutcome:
    status: PaperStatus
    quality_level: QualityLevel | None = None

    def __post_init__(self) -> None:
        if self.status not in {PaperStatus.READY, PaperStatus.PARTIALLY_READY}:
            raise ValueError("Successful processing outcome must be ready or partially_ready")


class ProcessingTaskStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def create(self, version_id: str) -> ProcessingTask:
        now = utc_now()
        with self.database.transaction() as connection:
            previous_attempt = connection.execute(
                "SELECT COALESCE(MAX(attempt),0) FROM processing_tasks WHERE version_id=?",
                (version_id,),
            ).fetchone()[0]
            task = ProcessingTask(
                task_id=str(uuid4()), version_id=version_id, status=TaskStatus.QUEUED,
                attempt=previous_attempt + 1, created_at=now, updated_at=now,
            )
            connection.execute(
                "INSERT INTO processing_tasks VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    task.task_id, task.version_id, task.status.value, task.attempt,
                    None, None, task.created_at, None, None, task.updated_at,
                ),
            )
        return task

    def get(self, task_id: str) -> ProcessingTask | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM processing_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def latest_for_version(self, version_id: str) -> ProcessingTask | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT * FROM processing_tasks WHERE version_id=?
                ORDER BY attempt DESC LIMIT 1""", (version_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def mark_running(self, task_id: str) -> ProcessingTask:
        now = utc_now()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """UPDATE processing_tasks SET status='running',started_at=?,updated_at=?
                WHERE task_id=? AND status='queued'""", (now, now, task_id)
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Task is not queued: {task_id}")
        return self.get(task_id)  # type: ignore[return-value]

    def mark_succeeded(self, task_id: str) -> ProcessingTask:
        return self._finish(task_id, TaskStatus.SUCCEEDED)

    def mark_failed(self, task_id: str, *, stage: str, message: str) -> ProcessingTask:
        return self._finish(task_id, TaskStatus.FAILED, stage=stage, message=message)

    def stale_running(self, older_than: datetime) -> tuple[ProcessingTask, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM processing_tasks WHERE status='running' AND updated_at < ?",
                (older_than.isoformat(),),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def _finish(
        self, task_id: str, status: TaskStatus, *, stage: str | None = None,
        message: str | None = None,
    ) -> ProcessingTask:
        now = utc_now()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """UPDATE processing_tasks SET status=?,error_stage=?,error_message=?,
                finished_at=?,updated_at=? WHERE task_id=? AND status IN ('queued','running')""",
                (status.value, stage, message, now, now, task_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Task is already finished: {task_id}")
        return self.get(task_id)  # type: ignore[return-value]

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ProcessingTask:
        return ProcessingTask(
            task_id=row["task_id"], version_id=row["version_id"],
            status=TaskStatus(row["status"]), attempt=row["attempt"],
            error_stage=row["error_stage"], error_message=row["error_message"],
            created_at=row["created_at"], started_at=row["started_at"],
            finished_at=row["finished_at"], updated_at=row["updated_at"],
        )


Processor = Callable[[str], ProcessingOutcome]


class ProcessingTaskRunner:
    def __init__(
        self,
        repository: SQLiteDomainRepository,
        store: ProcessingTaskStore,
        processor: Processor,
        *,
        max_workers: int = 1,
    ) -> None:
        self.repository = repository
        self.store = store
        self.processor = processor
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="paper-ingestion")
        self._lock = Lock()
        self._futures: dict[str, Future[ProcessingTask]] = {}

    def enqueue(self, version_id: str) -> ProcessingTask:
        task = self.store.create(version_id)
        with self._lock:
            self._futures[task.task_id] = self._executor.submit(self.run, task.task_id)
        return task

    def future(self, task_id: str) -> Future[ProcessingTask] | None:
        with self._lock:
            return self._futures.get(task_id)

    def retry(self, version_id: str) -> ProcessingTask:
        latest = self.store.latest_for_version(version_id)
        if latest is None or latest.status is not TaskStatus.FAILED:
            raise RuntimeError("Only a failed task can be retried")
        return self.enqueue(version_id)

    def run(self, task_id: str) -> ProcessingTask:
        task = self.store.mark_running(task_id)
        version = self.repository.get_version(task.version_id)
        if version is None:
            return self.store.mark_failed(task_id, stage="setup", message="Paper version not found")
        paper = self.repository.get_paper(version.paper_id)
        if paper is None:
            return self.store.mark_failed(task_id, stage="setup", message="Paper not found")

        self.repository.update_version(replace(version, parse_status=PaperStatus.PARSING))
        self.repository.update_paper(replace(paper, status=PaperStatus.PARSING))
        try:
            outcome = self.processor(version.version_id)
        except Exception as exc:
            self._set_failed(version.version_id, version.paper_id)
            return self.store.mark_failed(
                task_id, stage="processing", message=f"{type(exc).__name__}: {exc}"
            )

        current_version = self.repository.get_version(version.version_id)
        current_paper = self.repository.get_paper(version.paper_id)
        assert current_version is not None and current_paper is not None
        self.repository.update_version(replace(current_version, parse_status=outcome.status))
        self.repository.update_paper(
            replace(current_paper, status=outcome.status, quality_level=outcome.quality_level)
        )
        return self.store.mark_succeeded(task_id)

    def recover_stale(self, *, older_than: timedelta) -> tuple[str, ...]:
        threshold = datetime.now(UTC) - older_than
        recovered: list[str] = []
        for task in self.store.stale_running(threshold):
            version = self.repository.get_version(task.version_id)
            if version is not None:
                self._set_failed(version.version_id, version.paper_id)
            self.store.mark_failed(task.task_id, stage="recovery", message="Stale running task recovered")
            recovered.append(task.task_id)
        return tuple(recovered)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _set_failed(self, version_id: str, paper_id: str) -> None:
        version = self.repository.get_version(version_id)
        paper = self.repository.get_paper(paper_id)
        if version is not None:
            self.repository.update_version(replace(version, parse_status=PaperStatus.FAILED))
        if paper is not None:
            self.repository.update_paper(replace(paper, status=PaperStatus.FAILED))
