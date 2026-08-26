from paper_read_agent.application.sufficiency import AnswerPlan, SupportLevel
from paper_read_agent.application.verification import PostGenerationVerifier
from paper_read_agent.domain.evidence import Evidence, EvidenceRegistry
from paper_read_agent.domain.models import AnswerStatus
from paper_read_agent.llm.glm_client import Claim, StructuredAnswer


def ev(eid, paper="p"): return Evidence(eid,eid,paper,"Paper","v",1,1,(),"text",.9,"native")
class Judge:
    def __init__(self, values): self.values=values
    def verify(self, claim, evidence): return self.values.get(claim, SupportLevel.DIRECT)
def answer(*claims): return StructuredAnswer(AnswerStatus.ANSWERED,"answer",tuple(claims),(),(),(),None)
def plan(): return AnswerPlan(AnswerStatus.ANSWERED,(),())


def test_correct_and_partially_supported_claims_are_kept_or_removed():
    registry=EvidenceRegistry([ev("e1")], allowed_paper_ids=["p"])
    result=PostGenerationVerifier(Judge({"bad": SupportLevel.INSUFFICIENT})).verify(
        answer(Claim("good",("e1",),"direct"),Claim("bad",("e1",),"direct")), plan(), registry)
    assert [x.text for x in result.answer.claims] == ["good"]
    assert result.answer.answer_status is AnswerStatus.PARTIALLY_ANSWERED


def test_fake_and_out_of_scope_ids_never_display():
    registry=EvidenceRegistry([ev("e1")], allowed_paper_ids=[])
    result=PostGenerationVerifier(Judge({})).verify(
        answer(Claim("fake",("missing",),"direct"),Claim("scope",("e1",),"direct")), plan(), registry)
    assert result.severe_failure and result.answer.claims == ()
    assert result.answer.answer_status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_unmarked_inference_is_downgraded_and_hidden_conflict_exposed():
    registry=EvidenceRegistry([ev("e1")], allowed_paper_ids=["p"])
    result=PostGenerationVerifier(Judge({"infer": SupportLevel.INFERENCE,
                                         "conflict": SupportLevel.CONFLICT})).verify(
        answer(Claim("infer",("e1",),"direct"),Claim("conflict",("e1",),"direct")), plan(), registry)
    assert result.answer.claims[0].support == "inference"
    assert "conflict" in result.answer.conflicts
    assert result.answer.answer_status is AnswerStatus.CONFLICTED


def test_plan_unanswered_items_are_preserved():
    registry=EvidenceRegistry([ev("e1")], allowed_paper_ids=["p"])
    p=AnswerPlan(AnswerStatus.PARTIALLY_ANSWERED,(),("missing detail",))
    result=PostGenerationVerifier(Judge({})).verify(answer(Claim("ok",("e1",),"direct")),p,registry)
    assert result.answer.unanswered_items == ("missing detail",)
