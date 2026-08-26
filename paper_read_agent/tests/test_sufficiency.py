from paper_read_agent.application.sufficiency import SufficiencyChecker, SupportLevel
from paper_read_agent.domain.evidence import Evidence
from paper_read_agent.domain.models import AnswerStatus


def ev(eid, quality=.9, source="native_pdf"):
    return Evidence(eid, eid, "p", "Paper", "v", 1, 1, (), "text", quality, source)


class Judge:
    def __init__(self, values): self.values=values
    def judge(self, question, evidence): return self.values.get(question, {})


def test_complete_partial_and_no_evidence_plans():
    evidence=[ev("e1"), ev("e2")]
    complete=SufficiencyChecker(Judge({"a": {"e1": SupportLevel.DIRECT}})).check(["a"], evidence)
    partial=SufficiencyChecker(Judge({"a": {"e1": SupportLevel.DIRECT}, "b": {}})).check(["a","b"], evidence)
    none=SufficiencyChecker(Judge({})).check(["a"], evidence)
    assert complete.status is AnswerStatus.ANSWERED
    assert partial.status is AnswerStatus.PARTIALLY_ANSWERED and partial.unanswered_items == ("b",)
    assert none.status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_conflict_and_inference_are_explicit():
    evidence=[ev("e1"), ev("e2")]
    conflict=SufficiencyChecker(Judge({"q": {"e1": SupportLevel.CONFLICT, "e2": SupportLevel.CONFLICT}})).check(["q"], evidence)
    inference=SufficiencyChecker(Judge({"q": {"e1": SupportLevel.INFERENCE}})).check(["q"], evidence)
    assert conflict.status is AnswerStatus.CONFLICTED
    assert inference.items[0].support is SupportLevel.INFERENCE


def test_similar_but_unsupported_retrieval_is_not_treated_as_evidence():
    result=SufficiencyChecker(Judge({"q": {"e1": SupportLevel.INSUFFICIENT}})).check(["q"], [ev("e1")])
    assert result.items[0].evidence_ids == () and result.status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_low_quality_ocr_is_excluded_even_if_judge_calls_it_direct():
    evidence=[ev("ocr", .2, "rapidocr")]
    result=SufficiencyChecker(Judge({"q": {"ocr": SupportLevel.DIRECT}})).check(["q"], evidence)
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
