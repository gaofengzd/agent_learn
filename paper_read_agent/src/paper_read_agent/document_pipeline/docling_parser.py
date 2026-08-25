"""Docling PDF parser adapter.

Docling types are intentionally contained in this module. Downstream modules
consume the persisted dictionary or ``DoclingParseResult`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Protocol
import json

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from paper_read_agent.exceptions import DocumentProcessingError


class DoclingConversionError(DocumentProcessingError):
    pass


class Converter(Protocol):
    def convert(self, source: Path, **kwargs: object) -> Any: ...


@dataclass(frozen=True, slots=True)
class DoclingParseResult:
    version_id: str
    status: str
    parser_name: str
    parser_version: str
    json_path: Path
    metadata_path: Path
    document: dict[str, Any]
    parser_config: dict[str, Any]
    errors: tuple[str, ...]


class DoclingParser:
    def __init__(
        self,
        output_dir: Path,
        *,
        converter: Converter | None = None,
        enable_table_structure: bool = True,
        max_num_pages: int | None = None,
        max_file_size: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.max_num_pages = max_num_pages
        self.max_file_size = max_file_size
        self.enable_table_structure = enable_table_structure
        self.parser_version = package_version("docling")
        if converter is None:
            options = PdfPipelineOptions()
            options.do_ocr = False
            options.do_table_structure = enable_table_structure
            options.enable_remote_services = False
            self.converter: Converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
            )
        else:
            self.converter = converter

    def parse(self, pdf_path: str | Path, *, version_id: str) -> DoclingParseResult:
        source = Path(pdf_path)
        self._safe_artifact_name(version_id)
        kwargs: dict[str, object] = {"raises_on_error": False}
        if self.max_num_pages is not None:
            kwargs["max_num_pages"] = self.max_num_pages
        if self.max_file_size is not None:
            kwargs["max_file_size"] = self.max_file_size
        try:
            conversion = self.converter.convert(source, **kwargs)
        except Exception as exc:
            raise DoclingConversionError(
                f"Docling conversion failed: {type(exc).__name__}: {exc}"
            ) from exc

        status = self._status_value(conversion.status)
        errors = tuple(self._format_error(item) for item in getattr(conversion, "errors", ()))
        if status not in {
            ConversionStatus.SUCCESS.value,
            ConversionStatus.PARTIAL_SUCCESS.value,
        }:
            detail = "; ".join(errors) if errors else "no error detail"
            raise DoclingConversionError(f"Docling returned {status}: {detail}")
        document = getattr(conversion, "document", None)
        if document is None:
            raise DoclingConversionError("Docling returned no document")
        try:
            payload = document.export_to_dict()
        except Exception as exc:
            raise DoclingConversionError(
                f"Cannot export Docling document: {type(exc).__name__}: {exc}"
            ) from exc

        parser_config = {
            "do_ocr": False,
            "do_table_structure": self.enable_table_structure,
            "enable_remote_services": False,
            "max_num_pages": self.max_num_pages,
            "max_file_size": self.max_file_size,
        }
        destination = self._save_json(version_id, payload, suffix="docling.json")
        metadata_path = self._save_json(
            version_id,
            {
                "parser_name": "docling",
                "parser_version": self.parser_version,
                "status": status,
                "config": parser_config,
                "errors": list(errors),
            },
            suffix="docling.meta.json",
        )
        return DoclingParseResult(
            version_id=version_id,
            status=status,
            parser_name="docling",
            parser_version=self.parser_version,
            json_path=destination,
            metadata_path=metadata_path,
            document=payload,
            parser_config=parser_config,
            errors=errors,
        )

    def _save_json(self, version_id: str, payload: dict[str, Any], *, suffix: str) -> Path:
        safe_name = self._safe_artifact_name(version_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{safe_name}.{suffix}"
        temporary = self.output_dir / f".{safe_name}.{suffix}.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @staticmethod
    def _safe_artifact_name(version_id: str) -> str:
        safe_name = "".join(char for char in version_id if char.isalnum() or char in "-_")
        if not safe_name or safe_name != version_id:
            raise DoclingConversionError("Version ID is not safe for an artifact filename")
        return safe_name

    @staticmethod
    def _status_value(status: object) -> str:
        return str(getattr(status, "value", status))

    @staticmethod
    def _format_error(error: object) -> str:
        component = getattr(error, "component_type", None)
        message = getattr(error, "error_message", None)
        if message:
            prefix = f"{getattr(component, 'value', component)}: " if component else ""
            return prefix + str(message)
        return str(error)
