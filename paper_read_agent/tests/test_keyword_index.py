from pathlib import Path

import pytest

from paper_read_agent.domain.models import Chunk
from paper_read_agent.retrieval.keyword_index import SQLiteKeywordIndex
from paper_read_agent.retrieval.vector_index import VectorRecord


def record(chunk_id, text, *, paper="p1", version="v1", kind="text"):
    return VectorRecord(Chunk(chunk_id, version, None, text, len(text), 1, 1, (), kind), paper)


def test_recalls_chinese_english_models_abbreviations_numbers_and_formula(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "keywords.sqlite3")
    index.upsert([
        record("zh", "本文提出一种研究方法"), record("en", "retrieval augmented generation RAG"),
        record("model", "BGE-large-zh-v1.5 在 MTEB 得分 63.4"),
        record("formula", "公式 E=mc^2 描述能量"),
    ])
    assert index.query("方法")[0].chunk_id == "zh"
    assert index.query("RAG")[0].chunk_id == "en"
    assert index.query("bge-large-zh-v1.5")[0].chunk_id == "model"
    assert index.query("63.4")[0].chunk_id == "model"
    assert index.query("E=mc^2")[0].chunk_id == "formula"


def test_reference_scope_and_paper_filter_are_hard_constraints(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "keywords.sqlite3")
    index.upsert([record("body", "Transformer method"),
                  record("ref", "Transformer citation", kind="reference"),
                  record("other", "Transformer method", paper="p2")])
    assert {hit.chunk_id for hit in index.query("Transformer", paper_ids=["p1"])} == {"body"}
    assert {hit.chunk_id for hit in index.query("Transformer", paper_ids=["p1"],
                                                include_references=True)} == {"body", "ref"}


def test_delete_rebuild_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "keywords.sqlite3"
    index = SQLiteKeywordIndex(path)
    index.upsert([record("old", "old term"), record("keep", "keep term", paper="p2")])
    index.delete(paper_id="p1")
    assert index.query("old") == ()
    assert SQLiteKeywordIndex(path).query("keep")[0].chunk_id == "keep"
    index.rebuild([record("new", "new term")])
    assert index.query("keep") == ()
    assert index.query("new")[0].chunk_id == "new"


def test_upsert_replaces_searchable_text_without_residue(tmp_path: Path) -> None:
    index = SQLiteKeywordIndex(tmp_path / "keywords.sqlite3")
    index.upsert([record("same", "old phrase")])
    index.upsert([record("same", "new phrase")])
    assert index.query("old") == ()
    assert index.query("new")[0].chunk_id == "same"


def test_version_mismatch_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "keywords.sqlite3"
    SQLiteKeywordIndex(path, index_version="v1")
    with pytest.raises(RuntimeError, match="version mismatch"):
        SQLiteKeywordIndex(path, index_version="v2")
