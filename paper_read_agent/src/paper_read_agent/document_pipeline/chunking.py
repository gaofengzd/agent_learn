"""Structure-aware parent/child chunk construction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from transformers import AutoTokenizer

from paper_read_agent.config import ChunkSettings
from paper_read_agent.document_pipeline.normalizer import NormalizedDocument
from paper_read_agent.domain.models import Chunk, ContentBlock


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True, slots=True)
class ChunkBuildResult:
    version_id: str
    chunk_version: str
    chunks: tuple[Chunk, ...]
    adjacency: dict[str, tuple[str | None, str | None]]
    parameters: dict[str, int | float]


class ParentChildChunker:
    _SPECIAL_TYPES = {"table", "formula", "caption", "picture", "reference"}

    def __init__(
        self,
        settings: ChunkSettings,
        *,
        tokenizer: Tokenizer,
        chunk_version: str = "parent-child-v1",
    ) -> None:
        self.settings = settings
        self.tokenizer = tokenizer
        self.chunk_version = chunk_version

    @classmethod
    def from_local_model(
        cls, settings: ChunkSettings, model_path: str | Path,
        *, chunk_version: str = "parent-child-v1",
    ) -> "ParentChildChunker":
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Local tokenizer directory does not exist: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return cls(settings, tokenizer=tokenizer, chunk_version=chunk_version)

    def build(self, document: NormalizedDocument) -> ChunkBuildResult:
        page_numbers = {page.page_id: page.pdf_page_number for page in document.pages}
        child_specs: list[tuple[str, tuple[str, ...], str, tuple[str, ...], int, int]] = []
        pending: list[ContentBlock] = []

        def flush() -> None:
            if not pending:
                return
            child_specs.extend(self._ordinary_children(pending, page_numbers))
            pending.clear()

        for block in document.blocks:
            if block.block_type in self._SPECIAL_TYPES:
                flush()
                child_specs.extend(self._split_block(block, content_type=block.block_type,
                                                     page_numbers=page_numbers))
                continue
            if pending and (
                pending[-1].section_path != block.section_path
                or self._count("\n\n".join(item.text for item in pending + [block]))
                > self.settings.child_max_tokens
            ):
                flush()
            pending.append(block)
        flush()

        children = [self._make_child(document.version_id, index, spec) for index, spec in enumerate(child_specs)]
        parent_groups: list[list[Chunk]] = []
        current: list[Chunk] = []
        for child in children:
            proposed = "\n\n".join(item.text for item in current + [child])
            if current and (
                child.section_path != current[-1].section_path
                or child.content_type in self._SPECIAL_TYPES
                or current[-1].content_type in self._SPECIAL_TYPES
                or self._count(proposed) > self.settings.parent_max_tokens
            ):
                parent_groups.append(current)
                current = []
            current.append(child)
        if current:
            parent_groups.append(current)

        parents: list[Chunk] = []
        assigned_children: list[Chunk] = []
        for index, group in enumerate(parent_groups):
            text = "\n\n".join(item.text for item in group)
            parent_id = self._id("parent", document.version_id, self.chunk_version, str(index), text)
            parent = Chunk(
                chunk_id=parent_id, version_id=document.version_id, parent_chunk_id=None,
                text=text, token_count=self._count(text), page_start=min(item.page_start for item in group),
                page_end=max(item.page_end for item in group), section_path=group[0].section_path,
                content_type=group[0].content_type if len({item.content_type for item in group}) == 1 else "mixed",
                quality_score=self._mean(item.quality_score for item in group),
                index_version=self.chunk_version,
                block_ids=tuple(dict.fromkeys(block_id for item in group for block_id in item.block_ids)),
            )
            parents.append(parent)
            assigned_children.extend(
                Chunk(
                    chunk_id=item.chunk_id, version_id=item.version_id, parent_chunk_id=parent_id,
                    text=item.text, token_count=item.token_count, page_start=item.page_start,
                    page_end=item.page_end, section_path=item.section_path,
                    content_type=item.content_type, quality_score=item.quality_score,
                    index_version=item.index_version, block_ids=item.block_ids,
                )
                for item in group
            )
        adjacency = {
            item.chunk_id: (
                assigned_children[index - 1].chunk_id if index else None,
                assigned_children[index + 1].chunk_id if index + 1 < len(assigned_children) else None,
            )
            for index, item in enumerate(assigned_children)
        }
        return ChunkBuildResult(
            document.version_id, self.chunk_version, tuple(parents + assigned_children), adjacency,
            {
                "child_min_tokens": self.settings.child_min_tokens,
                "child_max_tokens": self.settings.child_max_tokens,
                "parent_min_tokens": self.settings.parent_min_tokens,
                "parent_max_tokens": self.settings.parent_max_tokens,
                "forced_split_overlap_ratio": self.settings.forced_split_overlap_ratio,
            },
        )

    def _ordinary_children(self, blocks: list[ContentBlock], page_numbers: dict[str, int]) -> list[tuple]:
        text = "\n\n".join(item.text for item in blocks)
        if self._count(text) <= self.settings.child_max_tokens:
            return [(text, blocks[0].section_path, "text", tuple(item.block_id for item in blocks),
                     page_numbers[blocks[0].page_id], page_numbers[blocks[-1].page_id],
                     self._mean(item.quality_score for item in blocks))]
        result: list[tuple] = []
        for block in blocks:
            result.extend(self._split_block(block, content_type="text", page_numbers=page_numbers))
        return result

    def _split_block(
        self, block: ContentBlock, *, content_type: str, page_numbers: dict[str, int]
    ) -> list[tuple]:
        tokens = self.tokenizer.encode(block.text, add_special_tokens=False)
        if len(tokens) <= self.settings.child_max_tokens:
            return [(block.text, block.section_path, content_type, (block.block_id,),
                     page_numbers[block.page_id], page_numbers[block.page_id], block.quality_score)]
        overlap = round(self.settings.child_max_tokens * self.settings.forced_split_overlap_ratio)
        step = max(self.settings.child_max_tokens - overlap, 1)
        return [
            (self.tokenizer.decode(tokens[start:start + self.settings.child_max_tokens], skip_special_tokens=True),
             block.section_path, content_type, (block.block_id,), page_numbers[block.page_id],
             page_numbers[block.page_id], block.quality_score)
            for start in range(0, len(tokens), step)
            if tokens[start:start + self.settings.child_max_tokens]
            and (start == 0 or len(tokens) - start > overlap)
        ]

    def _make_child(self, version_id: str, index: int, spec: tuple) -> Chunk:
        text, section, content_type, block_ids, page_start, page_end, quality_score = spec
        return Chunk(
            chunk_id=self._id("child", version_id, self.chunk_version, str(index), text),
            version_id=version_id, parent_chunk_id=None, text=text,
            token_count=self._count(text), page_start=page_start, page_end=page_end,
            section_path=section, content_type=content_type, index_version=self.chunk_version,
            quality_score=quality_score, block_ids=block_ids,
        )

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _id(*parts: str) -> str:
        return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _mean(values) -> float | None:
        present = [value for value in values if value is not None]
        return sum(present) / len(present) if present else None
