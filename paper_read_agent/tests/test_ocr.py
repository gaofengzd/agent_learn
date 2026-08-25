from pathlib import Path
from types import SimpleNamespace
import json

from paper_read_agent.document_pipeline.ocr import RapidOCRProcessor
from paper_read_agent.document_pipeline.preflight import PagePreflight, PDFPreflightReport


def page(number: int, needs_ocr: bool) -> PagePreflight:
    return PagePreflight(number, 0, 0.0, 0.0, needs_ocr, 600.0, 800.0)


def report(tmp_path: Path, *pages: PagePreflight) -> PDFPreflightReport:
    return PDFPreflightReport(tmp_path / "paper.pdf", len(pages), {}, tuple(pages), False, ())


class FakeEngine:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, image):
        self.calls.append(image)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


def output(texts=("方法",), scores=(0.95,), boxes=(((1, 2), (3, 2), (3, 4), (1, 4)),)):
    return SimpleNamespace(txts=texts, scores=scores, boxes=boxes)


def renderer(calls):
    def render(path, page_number, *, dpi):
        calls.append((Path(path), page_number, dpi))
        return f"page-{page_number}".encode()

    return render


def test_ocr_runs_only_for_selected_pages_and_persists_locations(tmp_path: Path) -> None:
    engine = FakeEngine([output(texts=("研究方法", "Method"), scores=(0.9, 0.8), boxes=(
        ((1, 2), (3, 2), (3, 4), (1, 4)), ((5, 6), (7, 6), (7, 8), (5, 8))))])
    render_calls = []
    processor = RapidOCRProcessor(tmp_path / "ocr", engine=engine)

    result = processor.process(
        tmp_path / "paper.pdf", report(tmp_path, page(1, False), page(2, True)),
        version_id="v1", render_page=renderer(render_calls)
    )

    assert [item.status for item in result.pages] == ["skipped", "success"]
    assert result.pages[0].source == "native_text"
    assert result.pages[1].source == "rapidocr"
    assert result.pages[1].lines[0].box[0] == (1.0, 2.0)
    assert render_calls == [(tmp_path / "paper.pdf", 2, 200)]
    saved = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert saved["pages"][1]["lines"][0]["text"] == "研究方法"


def test_marks_low_confidence_mixed_language_page(tmp_path: Path) -> None:
    processor = RapidOCRProcessor(tmp_path, engine=FakeEngine([output(("BERT模型",), (0.4,))]))
    result = processor.process(tmp_path / "x.pdf", report(tmp_path, page(1, True)),
                               version_id="v2", render_page=renderer([]))
    assert result.pages[0].low_confidence is True
    assert result.pages[0].mean_confidence == 0.4


def test_marks_blank_page_as_empty(tmp_path: Path) -> None:
    processor = RapidOCRProcessor(tmp_path, engine=FakeEngine([output((), (), ())]))
    result = processor.process(tmp_path / "x.pdf", report(tmp_path, page(1, True)),
                               version_id="v3", render_page=renderer([]))
    assert result.pages[0].status == "empty"
    assert result.pages[0].low_confidence is True


def test_records_page_level_engine_error_and_continues(tmp_path: Path) -> None:
    engine = FakeEngine([RuntimeError("bad image"), output(("下一页",), (0.9,))])
    processor = RapidOCRProcessor(tmp_path, engine=engine)
    result = processor.process(tmp_path / "x.pdf", report(tmp_path, page(1, True), page(2, True)),
                               version_id="v4", render_page=renderer([]))
    assert result.pages[0].status == "failed"
    assert result.pages[0].error == "RuntimeError: bad image"
    assert result.pages[1].status == "success"


def test_records_render_error_without_calling_engine(tmp_path: Path) -> None:
    engine = FakeEngine([])
    processor = RapidOCRProcessor(tmp_path, engine=engine)

    def broken(*args, **kwargs):
        raise ValueError("render failed")

    result = processor.process(tmp_path / "x.pdf", report(tmp_path, page(1, True)),
                               version_id="v5", render_page=broken)
    assert result.pages[0].status == "failed"
    assert "render failed" in result.pages[0].error
    assert engine.calls == []


def test_inconsistent_engine_output_is_a_page_error(tmp_path: Path) -> None:
    processor = RapidOCRProcessor(
        tmp_path, engine=FakeEngine([output(("text",), (), ())])
    )
    result = processor.process(tmp_path / "x.pdf", report(tmp_path, page(1, True)),
                               version_id="v6", render_page=renderer([]))
    assert result.pages[0].status == "failed"
    assert "inconsistent" in result.pages[0].error
