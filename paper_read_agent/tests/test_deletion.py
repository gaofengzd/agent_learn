from pathlib import Path
from paper_read_agent.application.deletion import PaperDeletionService
from paper_read_agent.application.processing_tasks import ProcessingTaskStore
from paper_read_agent.domain.models import Conversation,Message,MessageRole,Paper,PaperVersion
from paper_read_agent.persistence import SQLiteDatabase,SQLiteDomainRepository

class Index:
    def __init__(self,error=None):self.error=error;self.deleted=[]
    def delete(self,*,paper_id):
        if self.error:raise self.error
        self.deleted.append(paper_id)
def setup(tmp_path):
    pdf=tmp_path/"pdf";pdf.mkdir();file=pdf/"p.pdf";file.write_bytes(b"pdf")
    db=SQLiteDatabase(tmp_path/"db");db.initialize();r=SQLiteDomainRepository(db)
    r.create_paper(Paper("p","h","P",file_path=str(file)));r.create_version(PaperVersion("v","p","h"))
    r.create_conversation(Conversation("c","C","selected",("p",)))
    r.create_message(Message("m","c",MessageRole.ASSISTANT,"answer",retrieval_scope={"paper_ids":["p"]}))
    return pdf,file,db,r

def test_normal_delete_cleans_file_indexes_database_and_marks_history(tmp_path):
    pdf,file,db,r=setup(tmp_path);v=Index();k=Index()
    result=PaperDeletionService(r,ProcessingTaskStore(db),v,k,pdf).delete("p")
    assert result.deleted and not file.exists() and r.get_paper("p") is None and v.deleted==k.deleted==["p"]
    assert "p" in r.get_message("m").structured_payload["deleted_paper_ids"]

def test_processing_delete_is_rejected(tmp_path):
    pdf,file,db,r=setup(tmp_path);ProcessingTaskStore(db).create("v")
    try:PaperDeletionService(r,ProcessingTaskStore(db),Index(),Index(),pdf).delete("p")
    except RuntimeError as exc:assert "processing" in str(exc)
    else:raise AssertionError("active task must block")

def test_index_or_file_failure_is_reported_without_database_delete(tmp_path):
    pdf,file,db,r=setup(tmp_path)
    failed=PaperDeletionService(r,ProcessingTaskStore(db),Index(RuntimeError("down")),Index(),pdf).delete("p")
    assert not failed.deleted and r.get_paper("p") is not None and file.exists()
    failed2=PaperDeletionService(r,ProcessingTaskStore(db),Index(),Index(),pdf,file_remover=lambda p:(_ for _ in ()).throw(OSError("locked"))).delete("p")
    assert not failed2.deleted and any("file" in x for x in failed2.errors)

def test_path_outside_storage_and_other_paper_are_untouched(tmp_path):
    pdf,file,db,r=setup(tmp_path);outside=tmp_path/"outside.pdf";outside.write_text("x")
    r.create_paper(Paper("other","h2","O",file_path=str(outside)))
    try:PaperDeletionService(r,ProcessingTaskStore(db),Index(),Index(),pdf).delete("other")
    except ValueError:pass
    assert outside.exists() and r.get_paper("p") is not None
