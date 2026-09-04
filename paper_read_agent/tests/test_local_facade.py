import json
import time
from pathlib import Path
import fitz
from fastapi.testclient import TestClient

from paper_read_agent.application.processing_tasks import ProcessingOutcome
from paper_read_agent.config import AppSettings
from paper_read_agent.domain.models import PaperStatus,QualityLevel
from paper_read_agent.ui.local_facade import LocalUIFacade
from paper_read_agent.ui.web import create_app

def settings(tmp_path:Path)->AppSettings:
    model=tmp_path/"model";model.mkdir()
    return AppSettings.from_env({
      "PAPER_AGENT_DATA_DIR":str(tmp_path/"data"),"PAPER_AGENT_GLM_API_KEY":"test",
      "PAPER_AGENT_EMBEDDING_MODEL_PATH":str(model),"PAPER_AGENT_RERANKER_MODEL_PATH":str(model)},
      validate_model_paths=False)

def pdf_bytes()->bytes:
    document=fitz.open();page=document.new_page();page.insert_text((72,72),"Academic paper upload test")
    value=document.tobytes();document.close();return value

def test_default_runtime_facade_uploads_persists_enqueues_and_lists(tmp_path):
    facade=LocalUIFacade(settings(tmp_path),processor=lambda _:ProcessingOutcome(PaperStatus.READY,QualityLevel.READY))
    client=TestClient(create_app(facade))
    response=client.post("/library/upload",files={"files":("paper.pdf",pdf_bytes(),"application/pdf")})
    assert response.status_code==200 and "已接收 1 篇论文" in response.text
    deadline=time.time()+3
    while time.time()<deadline and facade.list_papers()[0]["status_label"]!="可用":time.sleep(.01)
    listed=facade.list_papers()[0]
    assert listed["title"]=="paper" and listed["status_label"]=="可用" and listed["version"]!="—"
    assert Path(facade.repository.list_papers()[0].file_path).read_bytes().startswith(b"%PDF-")
    facade.runner.shutdown()

def test_duplicate_upload_does_not_enqueue_second_task(tmp_path):
    facade=LocalUIFacade(settings(tmp_path),processor=lambda _:ProcessingOutcome(PaperStatus.READY,QualityLevel.READY))
    payload=pdf_bytes();files=[("paper.pdf","application/pdf",payload)]
    first=facade.upload_papers(files);second=facade.upload_papers(files)
    assert first[0]["duplicate"] is False and second[0]["duplicate"] is True
    with facade.database.connect() as connection:
        assert connection.execute("SELECT count(*) FROM processing_tasks").fetchone()[0]==1
    facade.runner.shutdown()

def test_failed_status_is_consistent_and_default_facade_can_delete(tmp_path):
    def fail(_):raise RuntimeError("parse failed")
    facade=LocalUIFacade(settings(tmp_path),processor=fail)
    result=facade.upload_papers([("paper.pdf","application/pdf",pdf_bytes())])[0]
    deadline=time.time()+3
    while time.time()<deadline and not facade.list_papers()[0]["can_retry"]:time.sleep(.01)
    listed=facade.list_papers()[0]
    assert listed["status_label"]=="处理失败" and listed["quality"]=="failed"
    assert listed["unreadable_label"]=="处理失败，暂不可阅读" and not listed["progress"]
    facade.delete_paper(result["paper_id"])
    assert facade.list_papers()==[]
    facade.runner.shutdown()


def test_analysis_result_is_restored_for_the_same_active_paper_version(tmp_path):
    configured=settings(tmp_path)
    first=LocalUIFacade(
        configured,
        processor=lambda _:ProcessingOutcome(PaperStatus.READY,QualityLevel.READY))
    first.upload_papers([("paper.pdf","application/pdf",pdf_bytes())])
    deadline=time.time()+3
    while time.time()<deadline and first.list_papers()[0]["status_label"]!="可用":
        time.sleep(.01)
    paper=first.repository.list_papers()[0]
    now="2026-01-01T00:00:00+00:00"
    payload={"analysis":{"summary":{"content":"持久化总结"}},
             "citations":[{"evidence_id":"ev1","label":"paper, p. 1",
                           "excerpt":"evidence","source_type":"native_pdf"}]}
    with first.database.transaction() as connection:
        connection.execute(
            "INSERT INTO analysis_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("analysis-1","summary",json.dumps([paper.paper_id]),
             json.dumps([paper.active_version_id]),"brief","succeeded","分析已完成",
             0,json.dumps(payload,ensure_ascii=False),now,now,now,now)
        )
    first.runner.shutdown()
    first._analysis_queue.shutdown()

    restored=LocalUIFacade(
        configured,
        processor=lambda _:ProcessingOutcome(PaperStatus.READY,QualityLevel.READY))
    view=restored.analysis_view()
    restored.runner.shutdown()
    restored._analysis_queue.shutdown()

    assert view["summary"]["content"]=="持久化总结"
    assert view["citations"][0]["evidence_id"]=="ev1"
