"""Page-selective RapidOCR adapter driven by PDF preflight results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Callable, Protocol
import json

from rapidocr import RapidOCR

from paper_read_agent.document_pipeline.preflight import PDFPreflightReport
from paper_read_agent.exceptions import DocumentProcessingError


class OCRProcessingError(DocumentProcessingError):
    pass


class OCREngine(Protocol):
    def __call__(self, image: bytes, **kwargs: object) -> Any: ...


@dataclass(frozen=True, slots=True)
class OCRLine:
    text: str
    confidence: float
    box: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class OCRPageResult:
    pdf_page_number: int
    status: str
    source: str
    lines: tuple[OCRLine, ...]
    mean_confidence: float | None
    low_confidence: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class OCRDocumentResult:
    version_id: str
    engine_name: str
    engine_version: str
    pages: tuple[OCRPageResult, ...]
    artifact_path: Path


class RapidOCRProcessor:
    def __init__(
        self,
        output_dir: Path,
        *,
        engine: OCREngine | None = None,
        renderer: Callable[[str | Path, int], bytes] | None = None,
        dpi: int = 200,
        low_confidence_threshold: float = 0.60,
    ) -> None:
        if dpi <= 0:
            raise ValueError("OCR render DPI must be positive")
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError("OCR confidence threshold must be between 0 and 1")
        self.output_dir = output_dir
        self.engine = engine or RapidOCR()
        self.renderer = renderer
        self.dpi = dpi
        self.low_confidence_threshold = low_confidence_threshold
        self.engine_version = package_version("rapidocr")

    def process(
        self,
        pdf_path: str | Path,
        report: PDFPreflightReport,
        *,
        version_id: str,
        render_page: Callable[..., bytes] | None = None,
    ) -> OCRDocumentResult:
        safe_name = self._safe_artifact_name(version_id)
        renderer = render_page or self.renderer
        if renderer is None:
            raise OCRProcessingError("A PDF page renderer is required")

        pages: list[OCRPageResult] = []
        for page in report.pages:
            if not page.needs_ocr:
                pages.append(self._skipped(page.pdf_page_number))
                continue
            try:
                image = renderer(pdf_path, page.pdf_page_number, dpi=self.dpi)
                raw = self.engine(image)
                pages.append(self._convert_page(page.pdf_page_number, raw))
            except Exception as exc:
                pages.append(
                    OCRPageResult(
                        pdf_page_number=page.pdf_page_number,
                        status="failed",
                        source="rapidocr",
                        lines=(),
                        mean_confidence=None,
                        low_confidence=True,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        artifact_path = self._save(safe_name, pages)
        return OCRDocumentResult(
            version_id=version_id,
            engine_name="rapidocr",
            engine_version=self.engine_version,
            pages=tuple(pages),
            artifact_path=artifact_path,
        )

    def _convert_page(self, page_number: int, raw: Any) -> OCRPageResult:
        texts = tuple(getattr(raw, "txts", None) or ())
        scores = tuple(getattr(raw, "scores", None) or ())
        boxes = tuple(getattr(raw, "boxes", None) or ())
        if not (len(texts) == len(scores) == len(boxes)):
            raise OCRProcessingError("RapidOCR returned inconsistent text, score, and box counts")
        lines = tuple(
            OCRLine(
                text=str(text),
                confidence=float(score),
                box=tuple((float(point[0]), float(point[1])) for point in box),
            )
            for text, score, box in zip(texts, scores, boxes, strict=True)
        )
        mean = sum(line.confidence for line in lines) / len(lines) if lines else None
        return OCRPageResult(
            pdf_page_number=page_number,
            status="success" if lines else "empty",
            source="rapidocr",
            lines=lines,
            mean_confidence=mean,
            low_confidence=mean is None or mean < self.low_confidence_threshold,
        )

    @staticmethod
    def _skipped(page_number: int) -> OCRPageResult:
        return OCRPageResult(
            pdf_page_number=page_number,
            status="skipped",
            source="native_text",
            lines=(),
            mean_confidence=None,
            low_confidence=False,
        )

    def _save(self, safe_name: str, pages: list[OCRPageResult]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{safe_name}.ocr.json"
        temporary = self.output_dir / f".{safe_name}.ocr.json.tmp"
        payload = {
            "engine_name": "rapidocr",
            "engine_version": self.engine_version,
            "dpi": self.dpi,
            "low_confidence_threshold": self.low_confidence_threshold,
            "pages": [asdict(page) for page in pages],
        }
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @staticmethod
    def _safe_artifact_name(version_id: str) -> str:
        safe_name = "".join(char for char in version_id if char.isalnum() or char in "-_")
        if not safe_name or safe_name != version_id:
            raise OCRProcessingError("Version ID is not safe for an artifact filename")
        return safe_name
