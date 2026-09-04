from pathlib import Path

import pytest

from paper_read_agent.config import AppSettings
from paper_read_agent.exceptions import ConfigurationError


def valid_env(tmp_path: Path) -> dict[str, str]:
    embedding = tmp_path / "embedding"
    reranker = tmp_path / "reranker"
    embedding.mkdir()
    reranker.mkdir()
    return {
        "PAPER_AGENT_GLM_API_KEY": "test-secret",
        "PAPER_AGENT_DATA_DIR": str(tmp_path / "runtime"),
        "PAPER_AGENT_EMBEDDING_MODEL_PATH": str(embedding),
        "PAPER_AGENT_RERANKER_MODEL_PATH": str(reranker),
    }


def test_loads_valid_configuration_and_defaults(tmp_path: Path) -> None:
    settings = AppSettings.from_env(valid_env(tmp_path))

    assert settings.models.glm_model == "glm-4.7"
    assert settings.chunks.child_min_tokens == 300
    assert settings.chunks.parent_max_tokens == 1500
    assert settings.retrieval.candidate_limit == 50
    assert settings.retrieval.rerank_input_limit == 24
    assert settings.retrieval.rerank_result_limit == 12
    assert settings.retrieval.evidence_context_ratio == 0.5
    assert settings.storage.pdf_dir == tmp_path / "runtime" / "pdfs"


def test_requires_glm_api_key(tmp_path: Path) -> None:
    env = valid_env(tmp_path)
    env.pop("PAPER_AGENT_GLM_API_KEY")

    with pytest.raises(ConfigurationError, match="GLM_API_KEY is required"):
        AppSettings.from_env(env)


def test_environment_values_override_defaults(tmp_path: Path) -> None:
    env = valid_env(tmp_path)
    env.update(
        {
            "PAPER_AGENT_GLM_MODEL": "configured-model",
            "PAPER_AGENT_CHILD_CHUNK_MIN_TOKENS": "320",
            "PAPER_AGENT_RETRIEVAL_CANDIDATE_LIMIT": "40",
            "PAPER_AGENT_RERANK_INPUT_LIMIT": "20",
            "PAPER_AGENT_RERANK_RESULT_LIMIT": "10",
            "PAPER_AGENT_EVIDENCE_CONTEXT_RATIO": "0.6",
            "PAPER_AGENT_LOG_LEVEL": "warning",
        }
    )

    settings = AppSettings.from_env(env)

    assert settings.models.glm_model == "configured-model"
    assert settings.chunks.child_min_tokens == 320
    assert settings.retrieval.candidate_limit == 40
    assert settings.retrieval.rerank_input_limit == 20
    assert settings.retrieval.rerank_result_limit == 10
    assert settings.retrieval.evidence_context_ratio == 0.6
    assert settings.runtime.log_level == "WARNING"


def test_rejects_missing_model_directory(tmp_path: Path) -> None:
    env = valid_env(tmp_path)
    env["PAPER_AGENT_EMBEDDING_MODEL_PATH"] = str(tmp_path / "missing")

    with pytest.raises(ConfigurationError, match="embedding model directory does not exist"):
        AppSettings.from_env(env)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PAPER_AGENT_CHILD_CHUNK_MIN_TOKENS", "invalid", "must be an integer"),
        ("PAPER_AGENT_FORCED_SPLIT_OVERLAP_RATIO", "1.0", "must be in"),
        ("PAPER_AGENT_EVIDENCE_CONTEXT_RATIO", "0.9", "between 0.4 and 0.6"),
        ("PAPER_AGENT_RERANK_INPUT_LIMIT", "60", "must not exceed candidate"),
        ("PAPER_AGENT_RERANK_RESULT_LIMIT", "25", "must not exceed rerank input"),
        ("PAPER_AGENT_LOG_LEVEL", "verbose", "Unsupported log level"),
    ],
)
def test_rejects_invalid_values(
    tmp_path: Path, name: str, value: str, message: str
) -> None:
    env = valid_env(tmp_path)
    env[name] = value

    with pytest.raises(ConfigurationError, match=message):
        AppSettings.from_env(env)


def test_runtime_directories_are_explicit_and_isolated(tmp_path: Path) -> None:
    settings = AppSettings.from_env(valid_env(tmp_path))

    settings.ensure_runtime_directories()

    assert settings.storage.data_dir.is_dir()
    assert settings.storage.pdf_dir.is_dir()
    assert settings.storage.parsed_dir.is_dir()
    assert settings.storage.chroma_dir.is_dir()
    assert settings.storage.log_dir.is_dir()
