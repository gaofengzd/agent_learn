"""Explainable parsing-quality assessment and availability classification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

from paper_read_agent.document_pipeline.normalizer import NormalizedDocument
from paper_read_agent.document_pipeline.ocr import OCRDocumentResult
from paper_read_agent.document_pipeline.preflight import PDFPreflightReport
from paper_read_agent.domain.models import Page, PaperStatus, QualityLevel, QualityReport
from paper_read_agent.persistence.repositories import SQLiteDomainRepository


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    ready_page_coverage: float = 0.90
    failed_page_coverage: float = 0.50
    max_ready_garbled_ratio: float = 0.10
    min_ready_ocr_confidence: float = 0.60

    def __post_init__(self) -> None:
        values = (
            self.ready_page_coverage, self.failed_page_coverage,
            self.max_ready_garbled_ratio, self.min_ready_ocr_confidence,
        )
        if any(not 0.0 <= item <= 1.0 for item in values):
            raise ValueError("Quality thresholds must be between 0 and 1")
        if self.failed_page_coverage > self.ready_page_coverage:
            raise ValueError("Failed coverage threshold cannot exceed ready threshold")


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    report: QualityReport
    pages: tuple[Page, ...]
    metrics: dict[str, int | float]


class ParseQualityEvaluator:
    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(
        self,
        document: NormalizedDocument,
        preflight: PDFPreflightReport,
        ocr: OCRDocumentResult,
    ) -> QualityAssessment:
        blocks_by_page = {page.page_id: [] for page in document.pages}
        for block in document.blocks:
            blocks_by_page.setdefault(block.page_id, []).append(block)
        ocr_by_page = {page.pdf_page_number: page for page in ocr.pages}
        preflight_by_page = {page.pdf_page_number: page for page in preflight.pages}
        missing = tuple(
            page.pdf_page_number for page in document.pages
            if not any(block.text.strip() for block in blocks_by_page.get(page.page_id, ()))
        )
        page_count = len(document.pages)
        coverage = (page_count - len(missing)) / page_count if page_count else 0.0
        garbled = (
            sum(page.garbled_ratio for page in preflight.pages) / len(preflight.pages)
            if preflight.pages else 0.0
        )
        ocr_scores = [
            page.mean_confidence for page in ocr.pages
            if page.status == "success" and page.mean_confidence is not None
        ]
        ocr_confidence = sum(ocr_scores) / len(ocr_scores) if ocr_scores else None
        warnings: list[str] = []
        if missing:
            warnings.append("Missing text on PDF pages: " + ", ".join(map(str, missing)))
        failed_ocr = [page.pdf_page_number for page in ocr.pages if page.status == "failed"]
        if failed_ocr:
            warnings.append("OCR failed on PDF pages: " + ", ".join(map(str, failed_ocr)))
        if garbled > self.thresholds.max_ready_garbled_ratio:
            warnings.append(f"Garbled character ratio is high: {garbled:.3f}")
        if ocr_confidence is not None and ocr_confidence < self.thresholds.min_ready_ocr_confidence:
            warnings.append(f"OCR confidence is low: {ocr_confidence:.3f}")
        heading_count = sum(block.block_type == "heading" for block in document.blocks)
        table_count = sum(block.block_type == "table" for block in document.blocks)
        formula_count = sum(block.block_type == "formula" for block in document.blocks)
        if document.blocks and heading_count == 0:
            warnings.append("No section structure was detected")

        if coverage < self.thresholds.failed_page_coverage or page_count == 0:
            level = QualityLevel.FAILED
        elif (
            coverage < self.thresholds.ready_page_coverage
            or garbled > self.thresholds.max_ready_garbled_ratio
            or (ocr_confidence is not None and ocr_confidence < self.thresholds.min_ready_ocr_confidence)
            or failed_ocr
        ):
            level = QualityLevel.PARTIALLY_READY
        else:
            level = QualityLevel.READY

        evaluated_pages = tuple(
            replace(
                page,
                quality_status=self._page_level(
                    page.pdf_page_number in missing,
                    preflight_by_page[page.pdf_page_number].garbled_ratio,
                    ocr_by_page.get(page.pdf_page_number),
                ),
            )
            for page in document.pages
        )
        report = QualityReport(
            quality_report_id=sha256(
                f"quality\x1f{document.version_id}".encode("utf-8")
            ).hexdigest()[:32],
            version_id=document.version_id,
            page_coverage=coverage,
            garbled_ratio=garbled,
            ocr_confidence=ocr_confidence,
            missing_pages=missing,
            warnings=tuple(warnings),
            final_level=level,
        )
        return QualityAssessment(
            report=report,
            pages=evaluated_pages,
            metrics={
                "page_count": page_count, "heading_count": heading_count,
                "table_count": table_count, "formula_count": formula_count,
            },
        )

    def persist(self, repository: SQLiteDomainRepository, assessment: QualityAssessment) -> None:
        version = repository.get_version(assessment.report.version_id)
        if version is None:
            raise KeyError(assessment.report.version_id)
        paper = repository.get_paper(version.paper_id)
        if paper is None:
            raise KeyError(version.paper_id)
        for page in assessment.pages:
            current = repository.get_page(page.page_id)
            if current is None:
                repository.create_page(page)
            else:
                repository.update_page(page)
        saved = repository.save_quality_report(assessment.report)
        status = {
            QualityLevel.READY: PaperStatus.READY,
            QualityLevel.PARTIALLY_READY: PaperStatus.PARTIALLY_READY,
            QualityLevel.FAILED: PaperStatus.FAILED,
        }[saved.final_level]
        repository.update_version(
            replace(version, parse_status=status, quality_report_id=saved.quality_report_id)
        )
        repository.update_paper(replace(paper, status=status, quality_level=saved.final_level))

    def _page_level(self, missing: bool, garbled: float, ocr_page: object | None) -> QualityLevel:
        if missing:
            return QualityLevel.FAILED
        if garbled > self.thresholds.max_ready_garbled_ratio:
            return QualityLevel.PARTIALLY_READY
        if ocr_page is not None and getattr(ocr_page, "status", None) == "failed":
            return QualityLevel.PARTIALLY_READY
        confidence = getattr(ocr_page, "mean_confidence", None)
        if confidence is not None and confidence < self.thresholds.min_ready_ocr_confidence:
            return QualityLevel.PARTIALLY_READY
        return QualityLevel.READY
