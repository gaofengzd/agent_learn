from fastapi.testclient import TestClient
from paper_read_agent.ui import create_app

class Facade:
    def __init__(self):self.actions=[];self.duplicate=False;self.error=None
    def navigation(self):return {"papers":[],"conversations":[]}
    def health(self):return []
    def list_papers(self):return [
        {"paper_id":"ready","title":"Ready","status_label":"可用","version":"v1","progress":"","quality":"ready","missing_pages":[],"error":"","readable":True,"can_retry":False},
        {"paper_id":"work","title":"Working","status_label":"处理中","version":"v2","progress":"解析 40%","quality":"partial","missing_pages":[3],"error":"","readable":False,"can_retry":False},
        {"paper_id":"bad","title":"Failed","status_label":"失败","version":"v3","progress":"","quality":"failed","missing_pages":[],"error":"OCR failed","readable":False,"can_retry":True}]
    def upload_papers(self,files):
        if self.error:raise self.error
        self.actions.append(("upload",len(files)));return [{"duplicate":self.duplicate} for _ in files]
    def delete_paper(self,pid):self.actions.append(("delete",pid))
    def retry_paper(self,pid):self.actions.append(("retry",pid))
    def reprocess_paper(self,pid):self.actions.append(("reprocess",pid))

def test_library_renders_all_states_quality_and_blocks_processing_read():
    text=TestClient(create_app(Facade())).get("/?section=library").text
    for value in ["Ready","Working","Failed","解析 40%","缺失页","OCR failed","处理中，暂不可阅读"]:assert value in text

def test_single_batch_duplicate_upload_and_backend_error():
    facade=Facade();client=TestClient(create_app(facade));facade.duplicate=True
    response=client.post("/library/upload",files=[("files",("a.pdf",b"%PDF","application/pdf")),("files",("b.pdf",b"%PDF","application/pdf"))])
    assert "已接收 2 篇论文；重复 2 篇" in response.text
    facade.error=RuntimeError("storage")
    assert "上传失败" in client.post("/library/upload",files={"files":("a.pdf",b"x","application/pdf")}).text

def test_delete_requires_confirmation_and_retry_reprocess_actions():
    facade=Facade();client=TestClient(create_app(facade))
    assert "必须确认删除" in client.post("/library/ready/delete",data={}).text
    client.post("/library/ready/delete",data={"confirm":"yes"});client.post("/library/bad/retry");client.post("/library/ready/reprocess")
    assert facade.actions==[("delete","ready"),("retry","bad"),("reprocess","ready")]
