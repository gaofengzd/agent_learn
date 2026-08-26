from pathlib import Path
from types import SimpleNamespace
import pytest
from paper_read_agent.application.qa_service import PreparedEvidence, QuestionAnsweringService
from paper_read_agent.application.sufficiency import AnswerPlan
from paper_read_agent.domain.evidence import Evidence, EvidenceRegistry
from paper_read_agent.domain.models import AnswerStatus, Conversation, Paper, PaperVersion
from paper_read_agent.exceptions import ModelInvocationError
from paper_read_agent.llm.glm_client import Claim, StructuredAnswer
from paper_read_agent.persistence import SQLiteDatabase, SQLiteDomainRepository
from paper_read_agent.retrieval.hybrid import HybridRecallResult
from paper_read_agent.retrieval.query_planner import QueryPlanner
from paper_read_agent.retrieval.reranker import RerankResult

class Retriever:
    def __init__(self, degraded=False): self.degraded=degraded
    def retrieve(self, plan): return HybridRecallResult((),(),self.degraded)
class Reranker:
    def rerank(self, plan, candidates): return RerankResult((),False)
class Preparer:
    def __init__(self,status=AnswerStatus.ANSWERED): self.status=status
    def prepare(self,plan,reranked):
        ev=Evidence("e1","c","p1","Paper","v1",1,1,("Methods",),"method",.9,"native")
        return PreparedEvidence(EvidenceRegistry([ev],allowed_paper_ids=plan.paper_ids),
                                AnswerPlan(self.status,(),()))
class GLM:
    def __init__(self,error=None): self.error=error
    def generate(self,messages):
        if self.error: raise self.error
        return StructuredAnswer(AnswerStatus.ANSWERED,"answer",(Claim("fact",("e1",),"direct"),),(),(),(),None)
class Verifier:
    def verify(self,answer,plan,registry): return SimpleNamespace(answer=answer)

@pytest.fixture
def repo(tmp_path):
    db=SQLiteDatabase(tmp_path/"db.sqlite3"); db.initialize(); r=SQLiteDomainRepository(db)
    r.create_paper(Paper("p1","h","Paper",file_path="x")); r.create_version(PaperVersion("v1","p1","h"))
    r.create_conversation(Conversation("conv","Chat","selected",("p1",))); return r,db

def service(repo,preparer=None,glm=None,retriever=None):
    return QuestionAnsweringService(repo,QueryPlanner(),retriever or Retriever(),Reranker(),
                                    preparer or Preparer(),glm or GLM(),Verifier())

def test_answer_citation_scope_and_atomic_persistence(repo):
    r,db=repo; result=service(r).answer("方法是什么",conversation_id="conv",scope_mode="selected",paper_ids=["p1"])
    assert result.answer.concise_answer == "answer" and result.citations[0].evidence_id == "e1"
    with db.connect() as c: assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2

def test_no_evidence_admits_unknown_without_model(repo):
    r,_=repo; result=service(r,Preparer(AnswerStatus.INSUFFICIENT_EVIDENCE)).answer(
        "unknown",conversation_id="conv",scope_mode="selected",paper_ids=["p1"])
    assert result.answer.answer_status is AnswerStatus.INSUFFICIENT_EVIDENCE and "不知道" in result.answer.concise_answer

def test_ambiguous_scope_returns_clarification(repo):
    r,_=repo; result=service(r).answer("这篇论文的方法",conversation_id="conv",scope_mode="library",paper_ids=["p1","p2"])
    assert result.answer.answer_status is AnswerStatus.OUT_OF_SCOPE

def test_model_failure_does_not_save_pseudo_success(repo):
    r,db=repo
    with pytest.raises(ModelInvocationError): service(r,glm=GLM(ModelInvocationError("down"))).answer(
        "方法",conversation_id="conv",scope_mode="selected",paper_ids=["p1"])
    with db.connect() as c: assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0

def test_degraded_retrieval_is_exposed(repo):
    r,_=repo; result=service(r,retriever=Retriever(True)).answer(
        "方法",conversation_id="conv",scope_mode="selected",paper_ids=["p1"])
    assert result.retrieval_degraded
