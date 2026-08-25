from pathlib import Path

import fitz
import pytest

from paper_read_agent.document_pipeline.preflight import (
    EncryptedPDFError,
    InvalidPDFError,
    PDFLimitError,
    PDFPreflight,
)


def create_pdf(path: Path, page_texts: list[str]) -> None:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_textbox(fitz.Rect(72, 72, 500, 700), text, fontsize=11)
    document.save(path)
    document.close()


def test_inspects_text_and_scan_like_pages(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdf"
    create_pdf(path, ["This is native academic text. " * 10, ""])

    report = PDFPreflight().inspect(path)

    assert report.page_count == 2
    assert report.pages[0].pdf_page_number == 1
    assert report.pages[0].text_char_count > 40
    assert report.pages[0].native_text_coverage > 0
    assert report.pages[0].needs_ocr is False
    assert report.pages[1].pdf_page_number == 2
    assert report.pages[1].needs_ocr is True


def test_renders_one_based_page_to_png(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    create_pdf(path, ["render me"])

    image = PDFPreflight().render_page(path, 1, dpi=72)

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="one-based"):
        PDFPreflight().render_page(path, 0)
    with pytest.raises(IndexError, match="out of range"):
        PDFPreflight().render_page(path, 2)


def test_rejects_corrupt_pdf(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not structurally valid")

    with pytest.raises(InvalidPDFError, match="Cannot open PDF"):
        PDFPreflight().inspect(path)


def test_rejects_password_protected_pdf(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "secret")
    document.save(
        path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    document.close()

    with pytest.raises(EncryptedPDFError, match="Password-protected"):
        PDFPreflight().inspect(path)


def test_enforces_optional_file_and_page_limits(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    create_pdf(path, ["one", "two"])

    with pytest.raises(PDFLimitError, match="file-size"):
        PDFPreflight(max_file_bytes=1).inspect(path)
    with pytest.raises(PDFLimitError, match="page-count"):
        PDFPreflight(max_pages=1).inspect(path)


def test_empty_page_is_valid_pdf_but_requires_ocr(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    create_pdf(path, [""])

    report = PDFPreflight().inspect(path)

    assert report.page_count == 1
    assert report.pages[0].text_char_count == 0
    assert report.pages[0].needs_ocr is True
