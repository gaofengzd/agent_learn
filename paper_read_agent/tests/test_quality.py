from dataclasses import replace
from pathlib import Path

import pytest

from paper_read_agent.document_pipeline.normalizer import NormalizedDocument
from paper_read_agent.document_pipeline.ocr import OCRDocumentResult, OCRPageResult
from paper_read_agent.document_pipeline.preflight import PagePreflight, PDFPreflightReport
from paper_read_agent.document_pipeline.quality import ParseQualityEvaluator, QualityThresholds
from paper_read_agent.domain.models import ContentBlock, Page, Paper, PaperStatus, PaperVersion, QualityLevel
from paper_read_agent.persistence import SQLiteDatabase, SQLiteDomainRepository


def fixture(tmp_path: Path, *, covered: int, total: int, garbled: float = 0.0,
            ocr_confidence: float | None = None, ocr_failed: bool = False):
    pages = tuple(Page(f"p{i}", "v1", i) for i in range(1, total + 1))
    blocks = tuple(
        ContentBlock(f"b{i}", "v1", f"p{i}", ("Methods",),
                     "heading" if i == 1 else "text", f"text {i}")
        for i in range(1, covered + 1)
    )
    normalized = NormalizedDocument("v1", pages, blocks)
    preflight_pages = tuple(PagePreflight(i, 10, 0.1, garbled, False, 1, 1) for i in range(1, total + 1))
    preflight = PDFPreflightReport(tmp_path / "x.pdf", total, {}, preflight_pages, False, ())
    ocr_pages = ()
    if ocr_confidence is not None or ocr_failed:
        ocr_pages = (OCRPageResult(1, "failed" if ocr_failed else "success", "rapidocr", (),
                                   ocr_confidence, False, "error" if ocr_failed else None),)
    ocr = OCRDocumentResult("v1", "rapidocr", "3", ocr_pages, tmp_path / "ocr.json")
    return normalized, preflight, ocr


@pytest.mark.parametrize(
    "covered,total,expected", [(10, 10, QualityLevel.READY), (8, 10, QualityLevel.PARTIALLY_READY),
                                (4, 10, QualityLevel.FAILED)]
)
def test_classifies_three_quality_levels(tmp_path: Path, covered, total, expected) -> None:
    assessment = ParseQualityEvaluator().evaluate(*fixture(tmp_path, covered=covered, total=total))
    assert assessment.report.final_level is expected
    assert assessment.report.page_coverage == covered / total


def test_threshold_boundaries_are_inclusive_for_better_level(tmp_path: Path) -> None:
    evaluator = ParseQualityEvaluator(QualityThresholds(0.8, 0.5, 0.1, 0.6))
    assessment = evaluator.evaluate(*fixture(tmp_path, covered=8, total=10, garbled=0.1,
                                              ocr_confidence=0.6))
    assert assessment.report.final_level is QualityLevel.READY


def test_low_ocr_garbled_missing_and_failure_warnings(tmp_path: Path) -> None:
    normalized, preflight, ocr = fixture(tmp_path, covered=1, total=2, garbled=0.2,
                                         ocr_confidence=0.4, ocr_failed=True)
    assessment = ParseQualityEvaluator().evaluate(normalized, preflight, ocr)
    assert assessment.report.final_level is QualityLevel.PARTIALLY_READY
    assert assessment.report.missing_pages == (2,)
    assert any("Garbled" in warning for warning in assessment.report.warnings)
    assert any("OCR failed" in warning for warning in assessment.report.warnings)
    assert assessment.pages[1].quality_status is QualityLevel.FAILED


def test_persist_updates_report_pages_and_failed_availability(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "db.sqlite3")
    database.initialize()
    repository = SQLiteDomainRepository(database)
    repository.create_paper(Paper("paper", "hash", "Paper", file_path="x.pdf"))
    repository.create_version(PaperVersion("v1", "paper", "hash"))
    assessment = ParseQualityEvaluator().evaluate(*fixture(tmp_path, covered=0, total=2))

    ParseQualityEvaluator().persist(repository, assessment)

    assert repository.get_version("v1").parse_status is PaperStatus.FAILED
    assert repository.get_paper("paper").quality_level is QualityLevel.FAILED
    assert repository.get_quality_report(assessment.report.quality_report_id).missing_pages == (1, 2)
    assert repository.get_page("p1").quality_status is QualityLevel.FAILED


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        QualityThresholds(ready_page_coverage=0.5, failed_page_coverage=0.6)
