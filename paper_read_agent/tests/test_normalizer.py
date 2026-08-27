from pathlib import Path

from paper_read_agent.document_pipeline.normalizer import DocumentNormalizer
from paper_read_agent.document_pipeline.ocr import OCRDocumentResult, OCRLine, OCRPageResult
from paper_read_agent.document_pipeline.preflight import PagePreflight, PDFPreflightReport


def inputs(tmp_path: Path):
    pages = tuple(PagePreflight(i, 50, 0.1, 0.0, i == 2, 600, 800) for i in (1, 2))
    preflight = PDFPreflightReport(tmp_path / "paper.pdf", 2, {}, pages, False, ())
    ocr = OCRDocumentResult(
        "v1", "rapidocr", "3.9.2",
        (
            OCRPageResult(1, "skipped", "native_text", (), None, False),
            OCRPageResult(2, "success", "rapidocr", (
                OCRLine("扫描补充", 0.8, ((1, 2), (5, 2), (5, 4), (1, 4))),
            ), 0.8, False),
        ),
        tmp_path / "ocr.json",
    )
    document = {
        "texts": [
            {"label": "page_header", "text": "Journal", "prov": [{"page_no": 1}]},
            {"label": "page_header", "text": "Journal", "prov": [{"page_no": 2}]},
            {"label": "section_header", "text": "Methods", "prov": [{"page_no": 1}]},
            {"label": "text", "text": "Native method", "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}]},
            {"label": "caption", "text": "Table 1", "prov": [{"page_no": 1}]},
            {"label": "formula", "text": "E=mc^2", "prov": [{"page_no": 1}]},
            {"label": "section_header", "text": "参考文献", "prov": [{"page_no": 2}]},
            {"label": "text", "text": "[1] Source", "prov": [{"page_no": 2}]},
        ],
        "tables": [{"data": {"table_cells": [{"text": "A"}, {"text": "B"}]}, "prov": [{"page_no": 1}]}],
        "pictures": [{"text": "Figure 1", "prov": [{"page_no": 1}]}],
    }
    return document, preflight, ocr


def test_normalizes_pages_sources_sections_and_references(tmp_path: Path) -> None:
    document, preflight, ocr = inputs(tmp_path)
    result = DocumentNormalizer().normalize(
        version_id="v1", docling_document=document, preflight=preflight, ocr=ocr
    )
    assert [page.pdf_page_number for page in result.pages] == [1, 2]
    assert result.pages[1].ocr_used is True
    assert all(block.text != "Journal" for block in result.blocks)
    assert next(block for block in result.blocks if block.text == "Native method").bbox == (1, 2, 3, 4)
    assert next(block for block in result.blocks if block.text == "扫描补充").source_type == "rapidocr"
    assert next(block for block in result.blocks if block.text == "[1] Source").block_type == "reference"


def test_ids_and_relations_are_stable(tmp_path: Path) -> None:
    document, preflight, ocr = inputs(tmp_path)
    normalizer = DocumentNormalizer()
    first = normalizer.normalize(version_id="v1", docling_document=document, preflight=preflight, ocr=ocr)
    second = normalizer.normalize(version_id="v1", docling_document=document, preflight=preflight, ocr=ocr)
    assert [item.page_id for item in first.pages] == [item.page_id for item in second.pages]
    assert [item.block_id for item in first.blocks] == [item.block_id for item in second.blocks]
    for index, block in enumerate(first.blocks):
        assert block.previous_block_id == (first.blocks[index - 1].block_id if index else None)
        assert block.next_block_id == (first.blocks[index + 1].block_id if index + 1 < len(first.blocks) else None)
    table = next(block for block in first.blocks if block.block_type == "table")
    assert table.related_block_ids
    assert {block.block_type for block in first.blocks} >= {"table", "formula", "picture", "caption"}


def test_repeated_provenance_blocks_have_stable_unique_ids(tmp_path: Path) -> None:
    document, preflight, ocr = inputs(tmp_path)
    document["texts"].append({"label":"text","text":"Repeated",
        "prov":[{"page_no":1,"bbox":[1,1,2,2]},{"page_no":1,"bbox":[3,3,4,4]}]})
    normalizer=DocumentNormalizer()
    first=normalizer.normalize(version_id="v1",docling_document=document,preflight=preflight,ocr=ocr)
    second=normalizer.normalize(version_id="v1",docling_document=document,preflight=preflight,ocr=ocr)
    repeated=[block.block_id for block in first.blocks if block.text=="Repeated"]
    assert len(repeated)==2 and len(set(repeated))==2
    assert repeated==[block.block_id for block in second.blocks if block.text=="Repeated"]


def test_deduplicates_ocr_text_already_present_on_same_page(tmp_path: Path) -> None:
    document, preflight, ocr = inputs(tmp_path)
    line = OCRLine("Native method", 0.9, ((1, 1), (2, 1), (2, 2), (1, 2)))
    duplicate_ocr = OCRDocumentResult(
        "v1", "rapidocr", "3.9.2",
        (ocr.pages[0], OCRPageResult(2, "success", "rapidocr", (line,), 0.9, False)),
        tmp_path / "ocr.json",
    )
    # Move the native text onto page 2 to exercise same-page deduplication.
    document["texts"][3]["prov"][0]["page_no"] = 2
    result = DocumentNormalizer().normalize(
        version_id="v1", docling_document=document, preflight=preflight, ocr=duplicate_ocr
    )
    assert sum(block.text == "Native method" for block in result.blocks) == 1


def test_rejects_mismatched_ocr_version(tmp_path: Path) -> None:
    document, preflight, ocr = inputs(tmp_path)
    wrong = OCRDocumentResult("other", ocr.engine_name, ocr.engine_version, ocr.pages, ocr.artifact_path)
    try:
        DocumentNormalizer().normalize(
            version_id="v1", docling_document=document, preflight=preflight, ocr=wrong
        )
    except ValueError as exc:
        assert "different paper version" in str(exc)
    else:
        raise AssertionError("version mismatch should fail")
