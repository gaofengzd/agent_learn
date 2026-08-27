"""Local BGE embeddings and persistent Chroma vector index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
import json

import chromadb
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from paper_read_agent.domain.models import Chunk
from paper_read_agent.exceptions import PaperAgentError


class VectorIndexError(PaperAgentError):
    pass


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class LocalBGEEmbedder:
    def __init__(self, model_path: str | Path, *, batch_size: int = 16,
                 max_length: int = 512) -> None:
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Local embedding model directory does not exist: {path}")
        self.path = path.resolve()
        self.batch_size = batch_size
        if max_length <= 0:raise ValueError("Embedding max length must be positive")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True)
        self.model.eval()
        self._dimension = int(self.model.config.hidden_size)

    @property
    def model_id(self) -> str:
        return f"bge-large-zh-v1.5@{self.path}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
            with torch.inference_mode():
                output = self.model(**encoded)
                embeddings = functional.normalize(output.last_hidden_state[:, 0], p=2, dim=1)
            vectors.extend(embeddings.cpu().tolist())
        return vectors


@dataclass(frozen=True, slots=True)
class VectorRecord:
    chunk: Chunk
    paper_id: str


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: str
    text: str
    distance: float
    metadata: dict[str, object]


class ChromaVectorIndex:
    def __init__(
        self,
        path: str | Path,
        embedder: Embedder,
        *,
        collection_name: str = "paper_chunks",
        index_version: str = "vector-v1",
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.collection_name = collection_name
        self.index_version = index_version
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={
                "embedding_model_id": embedder.model_id,
                "embedding_dimension": embedder.dimension,
                "index_version": index_version,
                "hnsw:space": "cosine",
            },
            embedding_function=None,
        )
        self._validate_metadata()

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        texts = [record.chunk.text for record in records]
        embeddings = self.embedder.embed(texts)
        self._validate_vectors(embeddings, len(records))
        self.collection.upsert(
            ids=[record.chunk.chunk_id for record in records],
            documents=texts,
            embeddings=embeddings,
            metadatas=[self._metadata(record) for record in records],
        )

    def query(
        self, text: str, *, limit: int = 50,
        paper_ids: Sequence[str] | None = None,
        version_ids: Sequence[str] | None = None,
        include_references: bool = False,
    ) -> tuple[VectorHit, ...]:
        if limit <= 0:
            raise ValueError("Vector query limit must be positive")
        vector = self.embedder.embed([text])
        self._validate_vectors(vector, 1)
        conditions: list[dict] = []
        if paper_ids:
            conditions.append({"paper_id": {"$in": list(paper_ids)}})
        if version_ids:
            conditions.append({"version_id": {"$in": list(version_ids)}})
        if not include_references:
            conditions.append({"content_type": {"$ne": "reference"}})
        where = conditions[0] if len(conditions) == 1 else ({"$and": conditions} if conditions else None)
        result = self.collection.query(
            query_embeddings=vector, n_results=limit, where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return tuple(
            VectorHit(chunk_id, document or "", float(distance), dict(metadata or {}))
            for chunk_id, document, metadata, distance
            in zip(ids, documents, metadatas, distances, strict=True)
        )

    def delete(self, *, paper_id: str | None = None, version_id: str | None = None) -> None:
        if (paper_id is None) == (version_id is None):
            raise ValueError("Specify exactly one paper_id or version_id")
        self.collection.delete(where={"paper_id" if paper_id else "version_id": paper_id or version_id})

    def rebuild(self, records: Sequence[VectorRecord]) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            self.collection_name,
            metadata={
                "embedding_model_id": self.embedder.model_id,
                "embedding_dimension": self.embedder.dimension,
                "index_version": self.index_version,
                "hnsw:space": "cosine",
            },
            embedding_function=None,
        )
        self.upsert(records)

    def _validate_metadata(self) -> None:
        metadata = self.collection.metadata or {}
        expected = {
            "embedding_model_id": self.embedder.model_id,
            "embedding_dimension": self.embedder.dimension,
            "index_version": self.index_version,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise VectorIndexError("Chroma index metadata mismatch: " + ", ".join(mismatches))

    def _validate_vectors(self, vectors: Sequence[Sequence[float]], expected_count: int) -> None:
        if len(vectors) != expected_count or any(len(item) != self.embedder.dimension for item in vectors):
            raise VectorIndexError("Embedding result count or dimension does not match index metadata")

    def _metadata(self, record: VectorRecord) -> dict[str, str | int | float]:
        chunk = record.chunk
        return {
            "paper_id": record.paper_id,
            "version_id": chunk.version_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_path": json.dumps(chunk.section_path, ensure_ascii=False),
            "content_type": chunk.content_type,
            "quality_score": chunk.quality_score if chunk.quality_score is not None else -1.0,
            "parent_chunk_id": chunk.parent_chunk_id or "",
            "block_ids": json.dumps(chunk.block_ids),
            "index_version": chunk.index_version or self.index_version,
        }
