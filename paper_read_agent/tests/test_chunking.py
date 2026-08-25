from paper_read_agent.config import ChunkSettings
from paper_read_agent.document_pipeline.chunking import ParentChildChunker
from paper_read_agent.document_pipeline.normalizer import NormalizedDocument
from paper_read_agent.domain.models import ContentBlock, Page


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, *, skip_special_tokens=True):
        return "".join(chr(item) for item in token_ids)


def settings(overlap=0.1):
    return ChunkSettings(3, 10, 10, 22, overlap)


def document(blocks):
    return NormalizedDocument("v1", (Page("opaque-a", "v1", 1), Page("opaque-b", "v1", 2)), tuple(blocks))


def block(number, text, *, page="opaque-a", kind="text", section=("Methods",)):
    return ContentBlock(f"b{number}", "v1", page, section, kind, text)


def children(result):
    return [item for item in result.chunks if item.parent_chunk_id is not None]


def test_groups_short_blocks_on_structure_boundaries() -> None:
    result = ParentChildChunker(settings(), tokenizer=CharacterTokenizer()).build(document([
        block(1, "abc"), block(2, "def"), block(3, "xyz", section=("Results",)),
    ]))
    values = children(result)
    assert [item.text for item in values] == ["abc\n\ndef", "xyz"]
    assert values[0].section_path == ("Methods",)
    assert values[1].section_path == ("Results",)


def test_forced_split_uses_configured_overlap_and_maximum() -> None:
    result = ParentChildChunker(settings(0.2), tokenizer=CharacterTokenizer()).build(
        document([block(1, "abcdefghijklmnopqrstuvw")])
    )
    values = children(result)
    assert [item.text for item in values] == ["abcdefghij", "ijklmnopqr", "qrstuvw"]
    assert all(item.token_count <= 10 for item in values)


def test_special_content_is_not_merged_with_ordinary_text() -> None:
    result = ParentChildChunker(settings(), tokenizer=CharacterTokenizer()).build(document([
        block(1, "intro"), block(2, "A|B", kind="table"), block(3, "E=mc2", kind="formula"),
        block(4, "[1] ref", kind="reference", page="opaque-b"),
    ]))
    values = children(result)
    assert [item.content_type for item in values] == ["text", "table", "formula", "reference"]
    assert values[-1].page_start == 2


def test_parent_child_and_adjacency_relations_are_complete() -> None:
    result = ParentChildChunker(settings(), tokenizer=CharacterTokenizer()).build(document([
        block(1, "abcdef"), block(2, "ghijkl"), block(3, "mnopqr"),
    ]))
    values = children(result)
    parent_ids = {item.chunk_id for item in result.chunks if item.parent_chunk_id is None}
    assert all(item.parent_chunk_id in parent_ids for item in values)
    assert result.adjacency[values[0].chunk_id][0] is None
    assert result.adjacency[values[-1].chunk_id][1] is None
    assert result.parameters["child_max_tokens"] == 10
    assert all(item.index_version == "parent-child-v1" for item in result.chunks)


def test_chunk_ids_and_snapshot_are_stable() -> None:
    chunker = ParentChildChunker(settings(), tokenizer=CharacterTokenizer())
    source = document([block(1, "中文abc"), block(2, "跨页", page="opaque-b")])
    first = chunker.build(source)
    second = chunker.build(source)
    assert [(item.chunk_id, item.text, item.page_start, item.page_end) for item in first.chunks] == [
        (item.chunk_id, item.text, item.page_start, item.page_end) for item in second.chunks
    ]


def test_local_tokenizer_loader_never_substitutes_missing_model(tmp_path) -> None:
    try:
        ParentChildChunker.from_local_model(settings(), tmp_path / "missing")
    except FileNotFoundError as exc:
        assert "Local tokenizer" in str(exc)
    else:
        raise AssertionError("missing local tokenizer should fail")
