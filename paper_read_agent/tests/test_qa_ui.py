from fastapi.testclient import TestClient
from paper_read_agent.ui import create_app
class Facade:
    def __init__(self):self.actions=[]
    def navigation(self):return {"papers":[],"conversations":[{"title":"Chat"}]}
    def health(self):return []
    def qa_view(self):return {"conversation_id":"c","title":"Chat","paper_ids":["p1"],"scope_label":"Paper A",
      "available_papers":[{"paper_id":"p1","title":"Paper A"},{"paper_id":"p2","title":"Paper B"}],
      "messages":[{"role_label":"用户","content":"q","answer_status":"","unanswered_items":[],"evidence_ids":[]},
                  {"role_label":"助手","content":"部分回答","answer_status":"部分回答","unanswered_items":["参数"],"evidence_ids":["e1"]}],
      "citations":[{"evidence_id":"e1","label":"Paper A, p. 2, Methods","excerpt":"original text","source_type":"native_pdf"}]}
    def create_session(self,title):self.actions.append(("create",title))
    def delete_session(self,cid):self.actions.append(("delete",cid))
    def change_scope(self,cid,mode,papers):self.actions.append(("scope",cid,mode,papers))
    def ask(self,cid,q):self.actions.append(("ask",cid,q))

def test_multiturn_partial_answer_scope_and_citation_expand_render():
    text=TestClient(create_app(Facade())).get("/?section=read").text
    for value in ["当前实际范围","部分回答","未回答","参数","Paper A, p. 2, Methods","original text","e1"]:assert value in text

def test_create_delete_scope_switch_and_ask_routes():
    f=Facade();c=TestClient(create_app(f));c.post("/sessions",data={"title":"New"});c.post("/sessions/c/scope",data={"scope_mode":"library","paper_ids":["p1","p2"]});c.post("/sessions/c/ask",data={"question":"why"});c.post("/sessions/c/delete")
    assert f.actions[0]==("create","New") and f.actions[-1]==("delete","c") and ("ask","c","why") in f.actions

def test_refusal_and_backend_error_are_visible():
    class Refusal(Facade):
        def qa_view(self):
            value=super().qa_view();value["messages"][-1].update(content="根据原文，我不知道",answer_status="证据不足");return value
        def ask(self,cid,q):raise RuntimeError("model")
    client=TestClient(create_app(Refusal()));assert "我不知道" in client.get("/?section=read").text
    assert "问答失败" in client.post("/sessions/c/ask",data={"question":"q"}).text
