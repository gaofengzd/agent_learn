from pathlib import Path

import pytest

from paper_read_agent.domain.models import Chunk
from paper_read_agent.retrieval.vector_index import (
    ChromaVectorIndex, LocalBGEEmbedder, VectorIndexError, VectorRecord,
)


class FakeEmbedder:
    def __init__(self, model_id="fake-v1", dimension=3):
        self.model_id = model_id
        self.dimension = dimension

    def embed(self, texts):
        return [[float("方法" in text or "method" in text.lower()),
                 float("结果" in text or "result" in text.lower()), 0.1] for text in texts]


def record(chunk_id, text, *, paper="p1", version="v1", kind="text"):
    return VectorRecord(Chunk(chunk_id, version, None, text, len(text), 1, 1,
                              ("Methods",), kind, index_version="vector-v1"), paper)


def test_persists_and_queries_chinese_and_english_with_filters(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    index = ChromaVectorIndex(path, FakeEmbedder())
    index.upsert([record("c1", "研究方法"), record("c2", "experimental method", paper="p2"),
                  record("c3", "[1] reference", kind="reference")])
    assert [hit.chunk_id for hit in index.query("方法", paper_ids=["p1"])] == ["c1"]
    assert [hit.chunk_id for hit in index.query("method", paper_ids=["p2"])] == ["c2"]
    assert "c3" not in {hit.chunk_id for hit in index.query("reference")}
    reopened = ChromaVectorIndex(path, FakeEmbedder())
    assert reopened.query("方法")[0].metadata["section_path"] == '["Methods"]'


def test_delete_and_rebuild_leave_no_stale_records(tmp_path: Path) -> None:
    index = ChromaVectorIndex(tmp_path / "chroma", FakeEmbedder())
    index.upsert([record("c1", "方法"), record("c2", "method", paper="p2", version="v2")])
    index.delete(version_id="v1")
    assert {hit.chunk_id for hit in index.query("方法", include_references=True)} == {"c2"}
    index.rebuild([record("c3", "结果")])
    assert {hit.chunk_id for hit in index.query("结果", include_references=True)} == {"c3"}


def test_rejects_model_or_dimension_mismatch_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    ChromaVectorIndex(path, FakeEmbedder())
    with pytest.raises(VectorIndexError, match="metadata mismatch"):
        ChromaVectorIndex(path, FakeEmbedder(model_id="other"))
    with pytest.raises(VectorIndexError, match="metadata mismatch"):
        ChromaVectorIndex(path, FakeEmbedder(dimension=4))


def test_rejects_bad_embedding_output(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    index = ChromaVectorIndex(tmp_path / "chroma", embedder)
    embedder.embed = lambda texts: [[1.0, 2.0]]
    with pytest.raises(VectorIndexError, match="dimension"):
        index.upsert([record("c1", "x")])


def test_missing_local_model_never_downloads(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Local embedding"):
        LocalBGEEmbedder(tmp_path / "missing")
