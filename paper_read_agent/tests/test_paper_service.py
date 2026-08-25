from io import BytesIO
from pathlib import Path

import pytest

from paper_read_agent.application import PaperService, UploadValidationError
from paper_read_agent.persistence import SQLiteDatabase, SQLiteDomainRepository


PDF = b"%PDF-1.7\nminimal-test-pdf\n%%EOF"


@pytest.fixture
def service(tmp_path: Path) -> PaperService:
    database = SQLiteDatabase(tmp_path / "database.sqlite3")
    database.initialize()
    return PaperService(SQLiteDomainRepository(database), tmp_path / "pdfs")


def upload(service: PaperService, filename: str = "paper.pdf", payload: bytes = PDF):
    return service.upload_pdf(
        original_filename=filename,
        content_type="application/pdf",
        stream=BytesIO(payload),
    )


def test_upload_stores_pdf_and_initial_records(service: PaperService) -> None:
    result = upload(service)

    assert result.duplicate is False
    assert result.version is not None
    assert result.paper.title == "paper"
    stored_path = Path(result.paper.file_path)
    assert stored_path.parent == service.pdf_dir
    assert stored_path.name == f"{result.paper.paper_id}.pdf"
    assert stored_path.read_bytes() == PDF
    assert service.repository.get_paper(result.paper.paper_id) == result.paper
    assert service.repository.get_version(result.version.version_id) == result.version


def test_duplicate_upload_returns_existing_paper_without_new_file(service: PaperService) -> None:
    first = upload(service, "first.pdf")
    second = upload(service, "renamed.pdf")

    assert second.duplicate is True
    assert second.paper.paper_id == first.paper.paper_id
    assert second.version is None
    assert len(list(service.pdf_dir.glob("*.pdf"))) == 1


def test_same_name_with_different_content_creates_distinct_papers(service: PaperService) -> None:
    first = upload(service, payload=PDF)
    second = upload(service, payload=b"%PDF-1.7\ndifferent\n%%EOF")

    assert first.paper.paper_id != second.paper.paper_id
    assert first.paper.content_hash != second.paper.content_hash
    assert len(list(service.pdf_dir.glob("*.pdf"))) == 2


@pytest.mark.parametrize(
    ("filename", "content_type", "payload", "message"),
    [
        ("paper.txt", "application/pdf", PDF, "Only .pdf"),
        ("paper.pdf", "text/plain", PDF, "MIME type"),
        ("paper.pdf", "application/pdf", b"", "empty"),
        ("paper.pdf", "application/pdf", b"not a pdf", "not a PDF"),
    ],
)
def test_rejects_invalid_uploads(
    service: PaperService,
    filename: str,
    content_type: str,
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(UploadValidationError, match=message):
        service.upload_pdf(
            original_filename=filename,
            content_type=content_type,
            stream=BytesIO(payload),
        )


def test_dangerous_original_filename_cannot_control_storage_path(service: PaperService) -> None:
    result = upload(service, "../../outside.pdf")

    assert Path(result.paper.file_path).parent == service.pdf_dir
    assert "outside" not in Path(result.paper.file_path).name


def test_repository_failure_rolls_back_file_and_paper_record(service: PaperService, monkeypatch) -> None:
    def fail_create_version(_version):
        raise RuntimeError("database failure")

    monkeypatch.setattr(service.repository, "create_version", fail_create_version)

    with pytest.raises(RuntimeError, match="database failure"):
        upload(service)

    assert not list(service.pdf_dir.iterdir())
    with service.repository.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM paper_versions").fetchone()[0] == 0
