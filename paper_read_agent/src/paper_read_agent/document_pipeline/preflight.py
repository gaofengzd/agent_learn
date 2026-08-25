"""Fast, read-only PDF preflight using PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata

import fitz

from paper_read_agent.exceptions import DocumentProcessingError


class InvalidPDFError(DocumentProcessingError):
    pass


class EncryptedPDFError(DocumentProcessingError):
    pass


class PDFLimitError(DocumentProcessingError):
    pass


@dataclass(frozen=True, slots=True)
class PagePreflight:
    pdf_page_number: int
    text_char_count: int
    native_text_coverage: float
    garbled_ratio: float
    needs_ocr: bool
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class PDFPreflightReport:
    path: Path
    page_count: int
    metadata: dict[str, str]
    pages: tuple[PagePreflight, ...]
    repaired_on_open: bool
    warnings: tuple[str, ...]


class PDFPreflight:
    def __init__(
        self,
        *,
        min_native_chars: int = 40,
        min_text_coverage: float = 0.002,
        max_garbled_ratio: float = 0.10,
        max_file_bytes: int | None = None,
        max_pages: int | None = None,
    ) -> None:
        self.min_native_chars = min_native_chars
        self.min_text_coverage = min_text_coverage
        self.max_garbled_ratio = max_garbled_ratio
        self.max_file_bytes = max_file_bytes
        self.max_pages = max_pages

    def inspect(self, path: str | Path) -> PDFPreflightReport:
        pdf_path = Path(path)
        if self.max_file_bytes is not None and pdf_path.stat().st_size > self.max_file_bytes:
            raise PDFLimitError("PDF exceeds configured file-size limit")
        try:
            document = fitz.open(pdf_path)
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise InvalidPDFError(f"Cannot open PDF: {type(exc).__name__}") from exc

        with document:
            if not document.is_pdf:
                raise InvalidPDFError("Document is not a PDF")
            if document.needs_pass or document.is_encrypted:
                raise EncryptedPDFError("Password-protected PDF is not supported")
            if document.page_count <= 0:
                raise InvalidPDFError("PDF has no pages")
            if self.max_pages is not None and document.page_count > self.max_pages:
                raise PDFLimitError("PDF exceeds configured page-count limit")

            warnings: list[str] = []
            if document.is_repaired:
                warnings.append("PDF was repaired while opening")
            pages = tuple(self._inspect_page(page) for page in document)
            metadata = {
                key: str(value)
                for key, value in (document.metadata or {}).items()
                if value not in (None, "")
            }
            return PDFPreflightReport(
                path=pdf_path,
                page_count=document.page_count,
                metadata=metadata,
                pages=pages,
                repaired_on_open=bool(document.is_repaired),
                warnings=tuple(warnings),
            )

    def render_page(self, path: str | Path, pdf_page_number: int, *, dpi: int = 200) -> bytes:
        if pdf_page_number <= 0:
            raise ValueError("PDF page number must be one-based and positive")
        try:
            with fitz.open(path) as document:
                if document.needs_pass or document.is_encrypted:
                    raise EncryptedPDFError("Password-protected PDF is not supported")
                if pdf_page_number > document.page_count:
                    raise IndexError("PDF page number is out of range")
                page = document.load_page(pdf_page_number - 1)
                pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
                return pixmap.tobytes("png")
        except (fitz.FileDataError, RuntimeError) as exc:
            raise InvalidPDFError(f"Cannot render PDF page: {type(exc).__name__}") from exc

    def _inspect_page(self, page: fitz.Page) -> PagePreflight:
        text = page.get_text("text", sort=True)
        character_count = sum(1 for char in text if not char.isspace())
        garbled_ratio = self._garbled_ratio(text)
        page_area = max(float(page.rect.get_area()), 1.0)
        text_area = 0.0
        for item in page.get_text("blocks", sort=True):
            if len(item) >= 7 and item[6] == 0:
                rectangle = fitz.Rect(item[:4]) & page.rect
                if not rectangle.is_empty:
                    text_area += float(rectangle.get_area())
        coverage = min(max(text_area / page_area, 0.0), 1.0)
        needs_ocr = (
            character_count < self.min_native_chars
            or coverage < self.min_text_coverage
            or garbled_ratio > self.max_garbled_ratio
        )
        return PagePreflight(
            pdf_page_number=page.number + 1,
            text_char_count=character_count,
            native_text_coverage=coverage,
            garbled_ratio=garbled_ratio,
            needs_ocr=needs_ocr,
            width=float(page.rect.width),
            height=float(page.rect.height),
        )

    @staticmethod
    def _garbled_ratio(text: str) -> float:
        characters = [char for char in text if not char.isspace()]
        if not characters:
            return 0.0
        garbled = sum(
            1
            for char in characters
            if char == "\ufffd" or unicodedata.category(char) in {"Cc", "Cs"}
        )
        return garbled / len(characters)
