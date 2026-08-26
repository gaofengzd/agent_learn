import threading
from paper_read_agent.application.reprocessing import ReprocessingService,VersionBuild
from paper_read_agent.domain.models import Paper,PaperStatus,PaperVersion
from paper_read_agent.persistence import SQLiteDatabase,SQLiteDomainRepository

class Processor:
    def __init__(self,error=None,embedding="embed",chunk="chunks"):self.error=error;self.embedding=embedding;self.chunk=chunk
    def process(self,version_id):
        if self.error:raise self.error
        return VersionBuild("docling","2","rapidocr","3",self.embedding,self.chunk)
def setup(tmp_path,processor):
    db=SQLiteDatabase(tmp_path/"db");db.initialize();r=SQLiteDomainRepository(db)
    r.create_paper(Paper("p","h","P",file_path="x",status=PaperStatus.READY,active_version_id="old"))
    r.create_version(PaperVersion("old","p","oldhash",parse_status=PaperStatus.READY))
    return r,ReprocessingService(r,processor,expected_embedding_model_id="embed",expected_chunk_version="chunks")

def test_success_atomically_switches_active_version(tmp_path):
    r,s=setup(tmp_path,Processor());result=s.reprocess("p")
    assert result.activated and r.get_paper("p").active_version_id==result.version_id
    assert r.get_version("old").parse_status is PaperStatus.READY

def test_parse_or_index_failure_keeps_old_version(tmp_path):
    for error in [RuntimeError("parse"),RuntimeError("index")]:
        r,s=setup(tmp_path/error.args[0],Processor(error));result=s.reprocess("p")
        assert not result.activated and r.get_paper("p").active_version_id=="old"
        assert r.get_version(result.version_id).parse_status is PaperStatus.FAILED

def test_model_or_chunk_version_mismatch_cannot_activate(tmp_path):
    for processor in [Processor(embedding="other"),Processor(chunk="other")]:
        r,s=setup(tmp_path/processor.embedding/processor.chunk,processor);result=s.reprocess("p")
        assert not result.activated and r.get_paper("p").active_version_id=="old"

def test_concurrent_reprocessing_is_rejected(tmp_path):
    entered=threading.Event();release=threading.Event()
    class Slow(Processor):
        def process(self,version_id):entered.set();release.wait(2);return super().process(version_id)
    r,s=setup(tmp_path,Slow());holder=[]
    thread=threading.Thread(target=lambda:holder.append(s.reprocess("p")));thread.start();entered.wait(1)
    try:s.reprocess("p")
    except RuntimeError as exc:assert "already active" in str(exc)
    else:raise AssertionError("concurrent call must fail")
    release.set();thread.join();assert holder[0].activated
