from io import BytesIO
import threading
import pytest
from paper_read_agent.application.paper_service import PaperService,UploadValidationError
from paper_read_agent.application.resource_limits import InputLimits,ResourceGuard
from paper_read_agent.llm.glm_client import GLMClient
from paper_read_agent.persistence import SQLiteDatabase,SQLiteDomainRepository

def test_empty_long_question_batch_and_invalid_limits():
    guard=ResourceGuard(InputLimits(max_question_chars=5,max_upload_batch=2))
    with pytest.raises(ValueError,match="empty"):guard.validate_question(" ")
    with pytest.raises(ValueError,match="length"):guard.validate_question("123456")
    with pytest.raises(ValueError,match="batch"):guard.validate_upload_batch([1,2,3])
    with pytest.raises(ValueError,match="positive"):InputLimits(max_pages=0)

def test_concurrent_task_limit_is_nonblocking():
    guard=ResourceGuard(InputLimits(max_concurrent_tasks=1));entered=threading.Event();release=threading.Event()
    def hold():
        with guard.task_slot():entered.set();release.wait(2)
    thread=threading.Thread(target=hold);thread.start();entered.wait(1)
    with pytest.raises(RuntimeError,match="Concurrent"): 
        with guard.task_slot():pass
    release.set();thread.join()

def test_oversized_pdf_is_rejected_before_storage(tmp_path):
    db=SQLiteDatabase(tmp_path/"db");db.initialize();service=PaperService(SQLiteDomainRepository(db),tmp_path/"pdf",max_file_bytes=8)
    with pytest.raises(UploadValidationError,match="file-size"):service.upload_pdf(
        original_filename="x.pdf",content_type="application/pdf",stream=BytesIO(b"%PDF-123456"))
    assert not (tmp_path/"pdf").exists()

def test_model_output_token_limit_is_enforced_without_transport_call():
    class Transport:
        def post(self,payload,timeout):raise AssertionError("must not call")
    with pytest.raises(ValueError,match="tokens"):GLMClient(Transport(),max_output_tokens=10).generate([],max_tokens=11)
