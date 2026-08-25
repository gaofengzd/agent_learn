from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from docling.datamodel.document import ConversionStatus

from paper_read_agent.document_pipeline.docling_parser import (
    DoclingConversionError,
    DoclingParser,
)


class FakeDocument:
    def __init__(self, payload):
        self.payload = payload

    def export_to_dict(self):
        return self.payload


class FakeConverter:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def convert(self, source, **kwargs):
        self.calls.append((source, kwargs))
        if self.error:
            raise self.error
        return self.result


def result(status=ConversionStatus.SUCCESS, payload=None, errors=()):
    document = FakeDocument(payload or {"name": "paper", "texts": [{"text": "method"}]})
    return SimpleNamespace(status=status, document=document, errors=errors)


def test_parses_and_atomically_persists_docling_json(tmp_path: Path) -> None:
    converter = FakeConverter(result())
    parser = DoclingParser(tmp_path / "parsed", converter=converter, max_num_pages=10)

    parsed = parser.parse(tmp_path / "paper.pdf", version_id="version-1")

    assert parsed.status == "success"
    assert parsed.parser_name == "docling"
    assert parsed.parser_version
    assert parsed.json_path.name == "version-1.docling.json"
    assert parsed.metadata_path.name == "version-1.docling.meta.json"
    assert json.loads(parsed.json_path.read_text(encoding="utf-8")) == parsed.document
    metadata = json.loads(parsed.metadata_path.read_text(encoding="utf-8"))
    assert metadata["parser_version"] == parsed.parser_version
    assert metadata["config"] == parsed.parser_config
    assert metadata["config"]["do_ocr"] is False
    assert metadata["config"]["do_table_structure"] is True
    assert not list(parsed.json_path.parent.glob("*.tmp"))
    assert converter.calls[0][1] == {"raises_on_error": False, "max_num_pages": 10}


def test_preserves_partial_success_and_errors(tmp_path: Path) -> None:
    error = SimpleNamespace(component_type=SimpleNamespace(value="backend"), error_message="page 2")
    parser = DoclingParser(
        tmp_path, converter=FakeConverter(result(ConversionStatus.PARTIAL_SUCCESS, errors=(error,)))
    )

    parsed = parser.parse(tmp_path / "paper.pdf", version_id="version-2")

    assert parsed.status == "partial_success"
    assert parsed.errors == ("backend: page 2",)


def test_rejects_failure_status_with_error_detail(tmp_path: Path) -> None:
    error = SimpleNamespace(error_message="invalid page", component_type=None)
    parser = DoclingParser(
        tmp_path, converter=FakeConverter(result(ConversionStatus.FAILURE, errors=(error,)))
    )

    with pytest.raises(DoclingConversionError, match="failure: invalid page"):
        parser.parse(tmp_path / "paper.pdf", version_id="version-3")


def test_wraps_converter_exception(tmp_path: Path) -> None:
    parser = DoclingParser(tmp_path, converter=FakeConverter(error=RuntimeError("broken")))

    with pytest.raises(DoclingConversionError, match="RuntimeError: broken"):
        parser.parse(tmp_path / "paper.pdf", version_id="version-4")


def test_rejects_unsafe_version_id(tmp_path: Path) -> None:
    converter = FakeConverter(result())
    parser = DoclingParser(tmp_path, converter=converter)

    with pytest.raises(DoclingConversionError, match="not safe"):
        parser.parse(tmp_path / "paper.pdf", version_id="../outside")

    assert converter.calls == []


@pytest.mark.parametrize(
    "payload, expected_key",
    [
        ({"texts": [{"text": "研究方法", "prov": [{"page_no": 1}]}]}, "texts"),
        ({"texts": [{"text": "Method", "prov": [{"page_no": 2}]}]}, "texts"),
        ({"groups": [{"name": "two-column", "children": []}]}, "groups"),
        ({"tables": [{"data": {"table_cells": []}, "prov": [{"page_no": 3}]}]}, "tables"),
        ({"texts": [{"label": "formula", "text": "E=mc^2"}]}, "texts"),
        ({"texts": [{"label": "section_header", "text": "Appendix A"}]}, "texts"),
    ],
)
def test_preserves_supported_docling_structures(
    tmp_path: Path, payload: dict, expected_key: str
) -> None:
    parser = DoclingParser(tmp_path, converter=FakeConverter(result(payload=payload)))

    parsed = parser.parse(tmp_path / "paper.pdf", version_id="fixture")

    assert parsed.document == payload
    assert expected_key in json.loads(parsed.json_path.read_text(encoding="utf-8"))
