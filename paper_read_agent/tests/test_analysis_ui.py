from fastapi.testclient import TestClient
from paper_read_agent.ui import create_app

class Facade:
    def __init__(self):self.actions=[]
    def navigation(self):return {"papers":[],"conversations":[]}
    def health(self):return []
    def analysis_view(self):return {
      "available_papers":[{"paper_id":"p1","title":"论文甲"},{"paper_id":"p2","title":"Paper B"}],
      "library_warning":"仅两篇论文，不能代表完整领域", "quality_warning":"论文乙 OCR 质量较低",
      "summary":{"level_label":"详细","scope_label":"论文甲、Paper B","content":"跨论文归纳", "evidence_ids":["e1"]},
      "methods":[{"field":"训练策略","role":"核心方法","status":"明示","content":"分阶段训练","evidence_ids":["e1"]},{"field":"消融设置","role":"评估方法","status":"未说明","content":"未说明","evidence_ids":[]}],
      "author_contributions":[{"status":"作者明示","content":"新框架","evidence_ids":["e1"]}],
      "agent_innovations":[{"status":"Agent 推断","content":"可能迁移至多模态","evidence_ids":["e2"]}],
      "citations":[{"evidence_id":"e1","label":"论文甲, p. 3, Methods","excerpt":"原文证据","source_type":"native_pdf"},{"evidence_id":"e2","label":"Paper B, p. 5","excerpt":"support","source_type":"ocr"}]}
    def summarize(self,papers,level):self.actions.append(("summary",papers,level))
    def extract_methods(self,papers):self.actions.append(("methods",papers))
    def analyze_innovations(self,papers):self.actions.append(("innovations",papers))

def test_three_reading_results_keep_provenance_and_warnings_visible():
    text=TestClient(create_app(Facade())).get("/?section=analysis").text
    for value in ["简要","标准","详细","跨论文归纳","核心方法","评估方法","明示","未说明","作者明示贡献","Agent 潜在创新","Agent 推断","资料库不足","质量警告","原文证据"]:assert value in text

def test_summary_method_and_innovation_routes_delegate_exactly():
    facade=Facade();client=TestClient(create_app(facade))
    client.post("/analysis/summary",data={"paper_ids":["p1","p2"],"level":"brief"})
    client.post("/analysis/methods",data={"paper_ids":["p1"]})
    client.post("/analysis/innovations",data={"paper_ids":["p2"]})
    assert facade.actions==[("summary",["p1","p2"],"brief"),("methods",["p1"]),("innovations",["p2"])]

def test_analysis_backend_error_is_not_hidden():
    class Broken(Facade):
        def summarize(self,papers,level):raise RuntimeError("glm")
    text=TestClient(create_app(Broken())).post("/analysis/summary",data={"paper_ids":["p1"],"level":"standard"}).text
    assert "分析失败" in text and "RuntimeError" in text

def test_background_analysis_submission_returns_without_running_inline():
    class Background(Facade):
        def submit_analysis(self,operation,papers,level):
            self.actions.append((operation,papers,level))
            return {"task_id":"t1","duplicate":False}
        def analysis_view(self):
            value=super().analysis_view()
            value.update(tasks=[{"task_id":"t1","operation":"summary","status":"running",
                                 "message":"正在检索、重排并生成"}],
                         analysis_active=True)
            return value
    facade=Background()
    response=TestClient(create_app(facade)).post(
        "/analysis/summary",data={"paper_ids":["p1"],"level":"detailed"})
    assert facade.actions==[("summary",["p1"],"detailed")]
    assert "阅读分析正在后台执行" in response.text
    assert "运行中" in response.text
    assert "取消任务" in response.text
    assert 'http-equiv="refresh"' in response.text

def test_duplicate_background_analysis_reports_existing_task():
    class Background(Facade):
        def submit_analysis(self,operation,papers,level):
            return {"task_id":"t1","duplicate":True}
    response=TestClient(create_app(Background())).post(
        "/analysis/summary",data={"paper_ids":["p1"],"level":"standard"})
    assert "相同的阅读分析正在执行，无需重复提交" in response.text

def test_cancel_analysis_route_delegates_task_id():
    class Background(Facade):
        def cancel_analysis(self,task_id):self.actions.append(("cancel",task_id))
    facade=Background()
    response=TestClient(create_app(facade)).post("/analysis/tasks/t1/cancel")
    assert facade.actions==[("cancel","t1")]
    assert "任务将在当前计算步骤结束后停止" in response.text
