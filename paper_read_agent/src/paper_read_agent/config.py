"""Typed application configuration loaded from environment variables.

This module intentionally uses only the Python standard library. Framework-
specific settings belong to later tickets after those technologies are chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import os

from paper_read_agent.exceptions import ConfigurationError


ENV_PREFIX = "PAPER_AGENT_"
DEFAULT_EMBEDDING_MODEL_PATH = Path(
    r"E:\00project\02agent\models\bge-large-zh-v1.5"
)
DEFAULT_RERANKER_MODEL_PATH = Path(
    r"E:\00project\02agent\models\bge-reranker-v2-m3"
)


def _read_int(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _read_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc


def _read_path(values: Mapping[str, str], name: str, default: Path) -> Path:
    raw = values.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True, slots=True)
class StorageSettings:
    data_dir: Path
    database_path: Path
    pdf_dir: Path
    parsed_dir: Path
    chroma_dir: Path
    log_dir: Path


@dataclass(frozen=True, slots=True)
class ModelSettings:
    glm_api_key: str
    glm_model: str
    embedding_model_path: Path
    reranker_model_path: Path


@dataclass(frozen=True, slots=True)
class ChunkSettings:
    child_min_tokens: int
    child_max_tokens: int
    parent_min_tokens: int
    parent_max_tokens: int
    forced_split_overlap_ratio: float


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    candidate_limit: int
    rerank_input_limit: int
    rerank_result_limit: int
    evidence_context_ratio: float


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    log_level: str


@dataclass(frozen=True, slots=True)
class AppSettings:
    storage: StorageSettings
    models: ModelSettings
    chunks: ChunkSettings
    retrieval: RetrievalSettings
    runtime: RuntimeSettings

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        require_glm_api_key: bool = True,
        validate_model_paths: bool = True,
    ) -> "AppSettings":
        values = os.environ if env is None else env
        data_dir = _read_path(values, f"{ENV_PREFIX}DATA_DIR", Path("data"))
        glm_api_key = values.get(f"{ENV_PREFIX}GLM_API_KEY", "").strip()

        settings = cls(
            storage=StorageSettings(
                data_dir=data_dir,
                database_path=_read_path(
                    values,
                    f"{ENV_PREFIX}DATABASE_PATH",
                    data_dir / "paper_agent.sqlite3",
                ),
                pdf_dir=data_dir / "pdfs",
                parsed_dir=data_dir / "parsed",
                chroma_dir=_read_path(
                    values, f"{ENV_PREFIX}CHROMA_DIR", data_dir / "chroma"
                ),
                log_dir=data_dir / "logs",
            ),
            models=ModelSettings(
                glm_api_key=glm_api_key,
                glm_model=values.get(f"{ENV_PREFIX}GLM_MODEL", "glm-4.7").strip(),
                embedding_model_path=_read_path(
                    values,
                    f"{ENV_PREFIX}EMBEDDING_MODEL_PATH",
                    DEFAULT_EMBEDDING_MODEL_PATH,
                ),
                reranker_model_path=_read_path(
                    values,
                    f"{ENV_PREFIX}RERANKER_MODEL_PATH",
                    DEFAULT_RERANKER_MODEL_PATH,
                ),
            ),
            chunks=ChunkSettings(
                child_min_tokens=_read_int(
                    values, f"{ENV_PREFIX}CHILD_CHUNK_MIN_TOKENS", 300
                ),
                child_max_tokens=_read_int(
                    values, f"{ENV_PREFIX}CHILD_CHUNK_MAX_TOKENS", 500
                ),
                parent_min_tokens=_read_int(
                    values, f"{ENV_PREFIX}PARENT_CHUNK_MIN_TOKENS", 1000
                ),
                parent_max_tokens=_read_int(
                    values, f"{ENV_PREFIX}PARENT_CHUNK_MAX_TOKENS", 1500
                ),
                forced_split_overlap_ratio=_read_float(
                    values, f"{ENV_PREFIX}FORCED_SPLIT_OVERLAP_RATIO", 0.15
                ),
            ),
            retrieval=RetrievalSettings(
                candidate_limit=_read_int(
                    values, f"{ENV_PREFIX}RETRIEVAL_CANDIDATE_LIMIT", 50
                ),
                rerank_input_limit=_read_int(
                    values, f"{ENV_PREFIX}RERANK_INPUT_LIMIT", 24
                ),
                rerank_result_limit=_read_int(
                    values, f"{ENV_PREFIX}RERANK_RESULT_LIMIT", 12
                ),
                evidence_context_ratio=_read_float(
                    values, f"{ENV_PREFIX}EVIDENCE_CONTEXT_RATIO", 0.50
                ),
            ),
            runtime=RuntimeSettings(
                log_level=values.get(f"{ENV_PREFIX}LOG_LEVEL", "INFO").upper()
            ),
        )
        settings.validate(
            require_glm_api_key=require_glm_api_key,
            validate_model_paths=validate_model_paths,
        )
        return settings

    def validate(
        self,
        *,
        require_glm_api_key: bool = True,
        validate_model_paths: bool = True,
    ) -> None:
        errors: list[str] = []
        if require_glm_api_key and not self.models.glm_api_key:
            errors.append(f"{ENV_PREFIX}GLM_API_KEY is required")
        if not self.models.glm_model:
            errors.append(f"{ENV_PREFIX}GLM_MODEL must not be empty")
        if validate_model_paths:
            for label, path in (
                ("embedding model", self.models.embedding_model_path),
                ("reranker model", self.models.reranker_model_path),
            ):
                if not path.is_dir():
                    errors.append(f"Configured {label} directory does not exist: {path}")

        chunk = self.chunks
        if min(
            chunk.child_min_tokens,
            chunk.child_max_tokens,
            chunk.parent_min_tokens,
            chunk.parent_max_tokens,
        ) <= 0:
            errors.append("Chunk token limits must be positive")
        if chunk.child_min_tokens > chunk.child_max_tokens:
            errors.append("Child chunk minimum must not exceed its maximum")
        if chunk.parent_min_tokens > chunk.parent_max_tokens:
            errors.append("Parent chunk minimum must not exceed its maximum")
        if chunk.child_max_tokens > chunk.parent_max_tokens:
            errors.append("Child chunk maximum must not exceed parent chunk maximum")
        if not 0 <= chunk.forced_split_overlap_ratio < 1:
            errors.append("Forced-split overlap ratio must be in [0, 1)")

        retrieval = self.retrieval
        if min(
            retrieval.candidate_limit,
            retrieval.rerank_input_limit,
            retrieval.rerank_result_limit,
        ) <= 0:
            errors.append("Retrieval limits must be positive")
        if retrieval.rerank_input_limit > retrieval.candidate_limit:
            errors.append("Rerank input limit must not exceed candidate limit")
        if retrieval.rerank_result_limit > retrieval.rerank_input_limit:
            errors.append("Rerank result limit must not exceed rerank input limit")
        if not 0.4 <= retrieval.evidence_context_ratio <= 0.6:
            errors.append("Evidence context ratio must be between 0.4 and 0.6")
        if self.runtime.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            errors.append(f"Unsupported log level: {self.runtime.log_level}")

        if errors:
            raise ConfigurationError("Invalid application configuration: " + "; ".join(errors))

    def ensure_runtime_directories(self) -> None:
        """Create only the configured application-owned runtime directories."""

        for path in (
            self.storage.data_dir,
            self.storage.pdf_dir,
            self.storage.parsed_dir,
            self.storage.chroma_dir,
            self.storage.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def secrets_for_redaction(self) -> tuple[str, ...]:
        return tuple(value for value in (self.models.glm_api_key,) if value)
