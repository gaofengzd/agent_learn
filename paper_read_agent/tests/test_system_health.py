from pathlib import Path

from paper_read_agent.application.system_health import HealthStatus, SystemHealthService
from paper_read_agent.config import AppSettings


def make_model(path: Path, *, hidden_size: int = 1024) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        f'{{"hidden_size": {hidden_size}, "architectures": ["TestModel"]}}', encoding="utf-8"
    )
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def settings(tmp_path: Path, *, api_key: str = "secret") -> AppSettings:
    embedding = tmp_path / "embedding"
    reranker = tmp_path / "reranker"
    make_model(embedding)
    make_model(reranker, hidden_size=768)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return AppSettings.from_env(
        {
            "PAPER_AGENT_GLM_API_KEY": api_key,
            "PAPER_AGENT_DATA_DIR": str(data_dir),
            "PAPER_AGENT_EMBEDDING_MODEL_PATH": str(embedding),
            "PAPER_AGENT_RERANKER_MODEL_PATH": str(reranker),
        },
        require_glm_api_key=bool(api_key),
    )


def test_health_report_detects_models_storage_and_safe_glm_details(tmp_path: Path, monkeypatch) -> None:
    value = settings(tmp_path)
    monkeypatch.setattr(
        "paper_read_agent.application.system_health.find_spec", lambda _module: object()
    )

    report = SystemHealthService(value).check()
    public = SystemHealthService.public_dict(report)

    by_name = {item.name: item for item in report.components}
    assert by_name["embedding_model"].status is HealthStatus.AVAILABLE
    assert by_name["embedding_model"].details["hidden_size"] == 1024
    assert by_name["reranker_model"].status is HealthStatus.AVAILABLE
    assert by_name["chroma_storage"].status is HealthStatus.AVAILABLE
    assert by_name["glm"].details == {"model": "glm-4.7", "api_key_configured": True}
    assert "secret" not in str(public)
    assert not value.storage.chroma_dir.exists()


def test_missing_model_path_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    value = settings(tmp_path)
    missing = tmp_path / "missing"
    object.__setattr__(value.models, "embedding_model_path", missing)
    monkeypatch.setattr(
        "paper_read_agent.application.system_health.find_spec", lambda _module: object()
    )

    report = SystemHealthService(value).check()
    item = next(item for item in report.components if item.name == "embedding_model")

    assert item.status is HealthStatus.UNAVAILABLE
    assert str(missing) in item.message


def test_incomplete_model_reports_missing_files(tmp_path: Path, monkeypatch) -> None:
    value = settings(tmp_path)
    (value.models.reranker_model_path / "model.safetensors").unlink()
    monkeypatch.setattr(
        "paper_read_agent.application.system_health.find_spec", lambda _module: object()
    )

    report = SystemHealthService(value).check()
    item = next(item for item in report.components if item.name == "reranker_model")

    assert item.status is HealthStatus.UNAVAILABLE
    assert "model weights" in item.details["missing"]


def test_missing_dependency_is_reported_without_installing(tmp_path: Path, monkeypatch) -> None:
    value = settings(tmp_path)
    monkeypatch.setattr(
        "paper_read_agent.application.system_health.find_spec",
        lambda module: None if module == "docling" else object(),
    )

    report = SystemHealthService(value).check()
    docling = next(item for item in report.components if item.name == "docling")

    assert docling.status is HealthStatus.UNAVAILABLE
    assert "not installed" in docling.message


def test_rapidocr_health_check_matches_the_runtime_adapter(tmp_path: Path, monkeypatch) -> None:
    value = settings(tmp_path)
    checked_modules = []

    def fake_find_spec(module: str):
        checked_modules.append(module)
        return object()

    monkeypatch.setattr(
        "paper_read_agent.application.system_health.find_spec", fake_find_spec
    )

    report = SystemHealthService(value).check()
    rapidocr = next(item for item in report.components if item.name == "rapidocr")

    assert rapidocr.status is HealthStatus.AVAILABLE
    assert rapidocr.details == {"module": "rapidocr"}
    assert "rapidocr_onnxruntime" not in checked_modules


def test_missing_glm_key_is_unavailable_and_not_exposed(tmp_path: Path, monkeypatch) -> None:
    value = settings(tmp_path, api_key="")
    monkeypatch.setattr(
        "paper_read_agent.application.system_health.find_spec", lambda _module: object()
    )

    report = SystemHealthService(value).check()
    glm = next(item for item in report.components if item.name == "glm")

    assert glm.status is HealthStatus.UNAVAILABLE
    assert glm.details == {"model": "glm-4.7"}
