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
