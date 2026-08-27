"""Normalize parser and OCR artifacts into stable domain pages and blocks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import re
from typing import Any, Iterable

from paper_read_agent.document_pipeline.ocr import OCRDocumentResult
from paper_read_agent.document_pipeline.preflight import PDFPreflightReport
from paper_read_agent.domain.models import ContentBlock, Page


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    version_id: str
    pages: tuple[Page, ...]
    blocks: tuple[ContentBlock, ...]


class DocumentNormalizer:
    _REFERENCE_TITLES = {"references", "bibliography", "参考文献"}
    _HEADER_LABELS = {"section_header", "title"}

    def normalize(
        self,
        *,
        version_id: str,
        docling_document: dict[str, Any],
        preflight: PDFPreflightReport,
        ocr: OCRDocumentResult,
    ) -> NormalizedDocument:
        if ocr.version_id != version_id:
            raise ValueError("OCR result belongs to a different paper version")
        ocr_by_page = {item.pdf_page_number: item for item in ocr.pages}
        pages = tuple(
            Page(
                page_id=self._id("page", version_id, str(item.pdf_page_number)),
                version_id=version_id,
                pdf_page_number=item.pdf_page_number,
                native_text_coverage=item.native_text_coverage,
                ocr_used=ocr_by_page.get(item.pdf_page_number) is not None
                and ocr_by_page[item.pdf_page_number].status in {"success", "empty", "failed"},
                ocr_confidence=(
                    ocr_by_page[item.pdf_page_number].mean_confidence
                    if item.pdf_page_number in ocr_by_page else None
                ),
            )
            for item in preflight.pages
        )
        page_ids = {item.pdf_page_number: item.page_id for item in pages}
        candidates = list(self._docling_candidates(docling_document))
        candidates = self._remove_repeated_margins(candidates)
        candidates.extend(self._ocr_candidates(ocr, candidates))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        section_path: tuple[str, ...] = ()
        in_references = False
        blocks: list[ContentBlock] = []
        for position, (page_number, order, label, text, bbox, source, quality) in enumerate(candidates):
            if page_number not in page_ids or not text.strip():
                continue
            normalized_label = self._block_type(label)
            if normalized_label == "heading":
                title = text.strip()
                section_path = (title,)
                in_references = title.casefold() in self._REFERENCE_TITLES
            block_type = "reference" if in_references and normalized_label != "heading" else normalized_label
            block_id = self._id(
                "block", version_id, str(page_number), str(order), str(position), block_type,
                re.sub(r"\s+", " ", text).strip(), source,
            )
            blocks.append(
                ContentBlock(
                    block_id=block_id,
                    version_id=version_id,
                    page_id=page_ids[page_number],
                    section_path=section_path,
                    block_type=block_type,
                    text=text.strip(),
                    bbox=bbox,
                    source_type=source,
                    quality_score=quality,
                )
            )
        blocks = self._link_neighbors(blocks)
        blocks = self._link_special_content(blocks)
        return NormalizedDocument(version_id, pages, tuple(blocks))

    def _docling_candidates(self, document: dict[str, Any]) -> Iterable[tuple]:
        order = 0
        for collection, fallback_label in (
            ("texts", "text"), ("tables", "table"), ("pictures", "picture")
        ):
            for item in document.get(collection, ()) or ():
                label = str(item.get("label") or fallback_label)
                text = self._item_text(item)
                provenances = item.get("prov") or ({"page_no": item.get("page_no", 1)},)
                for provenance in provenances:
                    page_number = int(provenance.get("page_no") or 1)
                    bbox = self._bbox(provenance.get("bbox") or item.get("bbox"))
                    yield (page_number, order, label, text, bbox, "native_pdf", None)
                order += 1

    def _ocr_candidates(self, ocr: OCRDocumentResult, native: list[tuple]) -> Iterable[tuple]:
        native_text = {
            (page, re.sub(r"\s+", "", text).casefold())
            for page, _, _, text, _, _, _ in native if text.strip()
        }
        for page in ocr.pages:
            if page.status != "success":
                continue
            for order, line in enumerate(page.lines, start=1_000_000):
                key = (page.pdf_page_number, re.sub(r"\s+", "", line.text).casefold())
                if key in native_text:
                    continue
                xs = [point[0] for point in line.box]
                ys = [point[1] for point in line.box]
                bbox = (min(xs), min(ys), max(xs), max(ys)) if xs and ys else None
                yield (
                    page.pdf_page_number, order, "text", line.text, bbox,
                    "rapidocr", line.confidence,
                )

    @staticmethod
    def _remove_repeated_margins(candidates: list[tuple]) -> list[tuple]:
        margin_labels = {"page_header", "page_footer"}
        counts = Counter(
            re.sub(r"\s+", " ", item[3]).strip().casefold()
            for item in candidates if item[2] in margin_labels and item[3].strip()
        )
        return [
            item for item in candidates
            if not (item[2] in margin_labels and counts[re.sub(r"\s+", " ", item[3]).strip().casefold()] > 1)
        ]

    @staticmethod
    def _link_neighbors(blocks: list[ContentBlock]) -> list[ContentBlock]:
        return [
            replace(
                block,
                previous_block_id=blocks[index - 1].block_id if index else None,
                next_block_id=blocks[index + 1].block_id if index + 1 < len(blocks) else None,
            )
            for index, block in enumerate(blocks)
        ]

    @staticmethod
    def _link_special_content(blocks: list[ContentBlock]) -> list[ContentBlock]:
        related: dict[str, set[str]] = {block.block_id: set() for block in blocks}
        special = {"table", "formula", "picture"}
        for index, block in enumerate(blocks):
            if block.block_type not in special:
                continue
            same_page = [
                (candidate_index, candidate)
                for candidate_index, candidate in enumerate(blocks)
                if candidate.page_id == block.page_id
                and candidate.block_id != block.block_id
                and candidate.block_type in {"caption", "text"}
            ]
            captions = [item for item in same_page if item[1].block_type == "caption"]
            pool = captions or same_page
            if pool:
                _, neighbor = min(pool, key=lambda item: abs(item[0] - index))
                related[block.block_id].add(neighbor.block_id)
                related[neighbor.block_id].add(block.block_id)
        return [replace(block, related_block_ids=tuple(sorted(related[block.block_id]))) for block in blocks]

    @staticmethod
    def _item_text(item: dict[str, Any]) -> str:
        if item.get("text") is not None:
            return str(item["text"])
        data = item.get("data") or {}
        cells = data.get("table_cells") or ()
        return "\n".join(str(cell.get("text", "")) for cell in cells if cell.get("text"))

    @staticmethod
    def _bbox(value: Any) -> tuple[float, float, float, float] | None:
        if not value:
            return None
        if isinstance(value, dict):
            keys = ("l", "t", "r", "b") if "l" in value else ("x0", "y0", "x1", "y1")
            if all(key in value for key in keys):
                return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]
        if len(value) == 4:
            return tuple(float(item) for item in value)  # type: ignore[return-value]
        return None

    @classmethod
    def _block_type(cls, label: str) -> str:
        mapping = {
            "section_header": "heading", "title": "heading", "list_item": "list",
            "table": "table", "formula": "formula", "picture": "picture",
            "caption": "caption", "page_header": "header", "page_footer": "footer",
        }
        return mapping.get(label, "text")

    @staticmethod
    def _id(*parts: str) -> str:
        return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
