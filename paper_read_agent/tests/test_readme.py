from pathlib import Path
from fastapi.testclient import TestClient
from paper_read_agent.ui.app import app

README=Path(__file__).parents[1]/"README.md"

def test_readme_covers_required_design_and_operations_without_a_secret():
    text=README.read_text(encoding="utf-8")
    for heading in ["数据流","Chunk 策略","检索方式","Prompt 与可信回答设计","已知失败案例","后续改进方向","隐私与安全边界"]:assert f"## {heading}" in text
    for value in ["bge-large-zh-v1.5","bge-reranker-v2-m3","glm-4.7","Chroma","RapidOCR","SQLite FTS5","uv sync --extra dev","uv run uvicorn"]:assert value in text
    assert "你的密钥" in text and "sk-" not in text

def test_documented_asgi_entrypoint_starts_and_names_all_pages():
    response=TestClient(app).get("/")
    assert response.status_code==200
    for page in ["论文库","问答","阅读分析"]:assert page in response.text

def test_readme_describes_default_upload_and_reading_wiring():
    text=README.read_text(encoding="utf-8")
    assert "创建 `LocalUIFacade`" in text and "真实上传论文" in text
    assert "问答与阅读分析会调用真实检索和 GLM 服务" in text
