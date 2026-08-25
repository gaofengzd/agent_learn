"""Paper upload validation, deduplication, and local file persistence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4
import os

from paper_read_agent.domain.models import Paper, PaperVersion
from paper_read_agent.exceptions import PaperAgentError
from paper_read_agent.persistence.repositories import SQLiteDomainRepository


PDF_SIGNATURE = b"%PDF-"


class UploadValidationError(PaperAgentError):
    """Raised when an uploaded file is not an acceptable PDF."""


@dataclass(frozen=True, slots=True)
class UploadResult:
    paper: Paper
    version: PaperVersion | None
    duplicate: bool


class PaperService:
    def __init__(self, repository: SQLiteDomainRepository, pdf_dir: Path) -> None:
        self.repository = repository
        self.pdf_dir = pdf_dir.resolve()

    def upload_pdf(
        self,
        *,
        original_filename: str,
        content_type: str | None,
        stream: BinaryIO,
    ) -> UploadResult:
        self._validate_metadata(original_filename, content_type)
        payload = stream.read()
        if not payload:
            raise UploadValidationError("Uploaded PDF is empty")
        if not payload.startswith(PDF_SIGNATURE):
            raise UploadValidationError("Uploaded content is not a PDF")

        content_hash = sha256(payload).hexdigest()
        if duplicate := self.repository.get_paper_by_content_hash(content_hash):
            return UploadResult(paper=duplicate, version=None, duplicate=True)

        paper_id = str(uuid4())
        version_id = str(uuid4())
        destination = (self.pdf_dir / f"{paper_id}.pdf").resolve()
        self._assert_inside_storage(destination)
        paper = Paper(
            paper_id=paper_id,
            content_hash=content_hash,
            title=Path(original_filename).stem,
            file_path=str(destination),
        )
        version = PaperVersion(
            version_id=version_id,
            paper_id=paper_id,
            content_hash=content_hash,
        )

        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        temp_path = (self.pdf_dir / f".{paper_id}.uploading").resolve()
        self._assert_inside_storage(temp_path)
        created_paper = False
        try:
            with temp_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            stored_paper = self.repository.create_paper(paper)
            created_paper = True
            stored_version = self.repository.create_version(version)
            temp_path.replace(destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            if created_paper:
                self.repository.delete_paper_record(paper_id)
            raise
        return UploadResult(paper=stored_paper, version=stored_version, duplicate=False)

    @staticmethod
    def _validate_metadata(filename: str, content_type: str | None) -> None:
        if Path(filename).suffix.lower() != ".pdf":
            raise UploadValidationError("Only .pdf files are accepted")
        if content_type and content_type.lower() != "application/pdf":
            raise UploadValidationError("Upload MIME type must be application/pdf")

    def _assert_inside_storage(self, path: Path) -> None:
        if path.parent != self.pdf_dir:
            raise UploadValidationError("Resolved upload path is outside PDF storage")
