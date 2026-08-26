from fastapi.testclient import TestClient
from paper_read_agent.ui import create_app

class Facade:
    def navigation(self):return {"papers":[{"title":"Paper A"}],"conversations":[{"title":"Chat 1"}]}
    def health(self):return [{"name":"Docling","status":"available","message":"可用"},{"name":"GLM","status":"warning","message":"未配置"}]

def test_layout_navigation_health_and_accessibility_render():
    response=TestClient(create_app(Facade())).get("/")
    assert response.status_code==200
    for text in ["论文库","问答","阅读分析","引用详情","Paper A","Chat 1","✓","△","跳到主要内容"]:assert text in response.text
    assert "{\"" not in response.text

def test_navigation_changes_main_heading():
    client=TestClient(create_app(Facade()))
    assert "论文库" in client.get("/?section=library").text
    assert "总结、方法与创新" in client.get("/?section=analysis").text

def test_backend_error_has_textual_alert_not_color_only():
    class Broken(Facade):
        def navigation(self):raise RuntimeError("db")
    text=TestClient(create_app(Broken())).get("/").text
    assert 'role="alert"' in text and "错误" in text and "RuntimeError" in text

def test_responsive_styles_include_narrow_layout():
    css=TestClient(create_app(Facade())).get("/static/styles.css")
    assert css.status_code==200 and "@media(max-width:850px)" in css.text and "grid-template-columns:1fr" in css.text

def test_default_shell_upload_reports_unconfigured_service_instead_of_500():
    client=TestClient(create_app(),raise_server_exceptions=False)
    response=client.post("/library/upload",files={"files":("paper.pdf",b"%PDF-1.4","application/pdf")})
    assert response.status_code==200
    assert "上传失败" in response.text and "论文处理服务尚未配置" in response.text
