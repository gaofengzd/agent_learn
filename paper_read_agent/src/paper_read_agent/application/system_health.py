"""Read-only health checks for configured models and external components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec
from pathlib import Path
import json
import os

from paper_read_agent.config import AppSettings


class HealthStatus(StrEnum):
    AVAILABLE = "available"
    WARNING = "warning"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str
    details: dict[str, object]


@dataclass(frozen=True, slots=True)
class SystemHealthReport:
    components: tuple[ComponentHealth, ...]

    @property
    def status(self) -> HealthStatus:
        statuses = {item.status for item in self.components}
        if HealthStatus.UNAVAILABLE in statuses:
            return HealthStatus.UNAVAILABLE
        if HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        return HealthStatus.AVAILABLE


class SystemHealthService:
    REQUIRED_DEPENDENCIES = {
        "docling": "Docling PDF parser",
        "rapidocr": "RapidOCR engine",
        "fitz": "PyMuPDF",
        "chromadb": "Chroma",
    }

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def check(self) -> SystemHealthReport:
        components = [
            self._check_model("embedding_model", self.settings.models.embedding_model_path),
            self._check_model("reranker_model", self.settings.models.reranker_model_path),
            self._check_chroma_path(),
            self._check_glm_configuration(),
        ]
        components.extend(
            self._check_dependency(module, label)
            for module, label in self.REQUIRED_DEPENDENCIES.items()
        )
        return SystemHealthReport(tuple(components))

    @staticmethod
    def public_dict(report: SystemHealthReport) -> dict[str, object]:
        return {
            "status": report.status.value,
            "components": [
                {
                    "name": item.name,
                    "status": item.status.value,
                    "message": item.message,
                    "details": item.details,
                }
                for item in report.components
            ],
        }

    def _check_model(self, name: str, path: Path) -> ComponentHealth:
        required = ("config.json", "tokenizer_config.json")
        missing = [filename for filename in required if not (path / filename).is_file()]
        has_weights = any(
            (path / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        if not path.is_dir():
            return ComponentHealth(
                name, HealthStatus.UNAVAILABLE, f"Model directory does not exist: {path}",
                {"path": str(path)},
            )
        if missing or not has_weights:
            if not has_weights:
                missing.append("model weights")
            return ComponentHealth(
                name, HealthStatus.UNAVAILABLE, "Model directory is incomplete",
                {"path": str(path), "missing": missing},
            )
        details: dict[str, object] = {"path": str(path)}
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if "hidden_size" in config:
                details["hidden_size"] = config["hidden_size"]
            if "architectures" in config:
                details["architectures"] = config["architectures"]
        except (OSError, ValueError) as exc:
            return ComponentHealth(
                name, HealthStatus.WARNING, "Model exists but config.json cannot be read",
                {"path": str(path), "error": type(exc).__name__},
            )
        return ComponentHealth(name, HealthStatus.AVAILABLE, "Model files are present", details)

    def _check_chroma_path(self) -> ComponentHealth:
        path = self.settings.storage.chroma_dir
        check_path = path if path.exists() else path.parent
        if not check_path.exists():
            return ComponentHealth(
                "chroma_storage", HealthStatus.WARNING,
                "Chroma path and parent do not exist yet", {"path": str(path)},
            )
        if not check_path.is_dir() or not os.access(check_path, os.W_OK):
            return ComponentHealth(
                "chroma_storage", HealthStatus.UNAVAILABLE,
                "Chroma storage location is not a writable directory", {"path": str(path)},
            )
        return ComponentHealth(
            "chroma_storage", HealthStatus.AVAILABLE,
            "Chroma storage location is writable", {"path": str(path)},
        )
    def _check_glm_configuration(self) -> ComponentHealth:
        if not self.settings.models.glm_api_key:
            return ComponentHealth(
                "glm", HealthStatus.UNAVAILABLE, "GLM API key is not configured",
                {"model": self.settings.models.glm_model},
            )
        return ComponentHealth(
            "glm", HealthStatus.AVAILABLE, "GLM configuration is present",
            {"model": self.settings.models.glm_model, "api_key_configured": True},
        )

    @staticmethod
    def _check_dependency(module: str, label: str) -> ComponentHealth:
        if find_spec(module) is None:
            return ComponentHealth(
                module, HealthStatus.UNAVAILABLE, f"{label} dependency is not installed",
                {"module": module},
            )
        return ComponentHealth(
            module, HealthStatus.AVAILABLE, f"{label} dependency is importable",
            {"module": module},
        )
