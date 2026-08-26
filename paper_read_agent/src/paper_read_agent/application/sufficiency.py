"""Pre-generation evidence sufficiency planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence

from paper_read_agent.domain.evidence import Evidence
from paper_read_agent.domain.models import AnswerStatus


class SupportLevel(StrEnum):
    DIRECT = "direct"
    INFERENCE = "inference"
    CONFLICT = "conflict"
    INSUFFICIENT = "insufficient"


class SupportJudge(Protocol):
    def judge(self, question: str, evidence: Sequence[Evidence]) -> dict[str, SupportLevel]: ...


@dataclass(frozen=True, slots=True)
class AnswerPlanItem:
    question: str
    support: SupportLevel
    evidence_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class AnswerPlan:
    status: AnswerStatus
    items: tuple[AnswerPlanItem, ...]
    unanswered_items: tuple[str, ...]


class SufficiencyChecker:
    def __init__(self, judge: SupportJudge, *, min_evidence_quality: float = 0.5) -> None:
        self.judge, self.min_evidence_quality = judge, min_evidence_quality

    def check(self, subquestions: Sequence[str], evidence: Sequence[Evidence]) -> AnswerPlan:
        items = []
        for question in dict.fromkeys(subquestions):
            judgments = self.judge.judge(question, evidence)
            valid = [(item, judgments.get(item.evidence_id, SupportLevel.INSUFFICIENT))
                     for item in evidence
                     if item.quality_score is None or item.quality_score >= self.min_evidence_quality]
            direct = [item.evidence_id for item, level in valid if level is SupportLevel.DIRECT]
            inferred = [item.evidence_id for item, level in valid if level is SupportLevel.INFERENCE]
            conflicts = [item.evidence_id for item, level in valid if level is SupportLevel.CONFLICT]
            if conflicts:
                level, ids, reason = SupportLevel.CONFLICT, tuple(conflicts), "Available evidence conflicts"
            elif direct:
                level, ids, reason = SupportLevel.DIRECT, tuple(direct), "Direct evidence is available"
            elif inferred:
                level, ids, reason = SupportLevel.INFERENCE, tuple(inferred), "Only limited inference is possible"
            else:
                level, ids, reason = SupportLevel.INSUFFICIENT, (), "No sufficiently strong evidence"
            items.append(AnswerPlanItem(question, level, ids, reason))
        unanswered = tuple(item.question for item in items if item.support is SupportLevel.INSUFFICIENT)
        answerable = len(items) - len(unanswered)
        if not items or answerable == 0: status = AnswerStatus.INSUFFICIENT_EVIDENCE
        elif unanswered: status = AnswerStatus.PARTIALLY_ANSWERED
        elif any(item.support is SupportLevel.CONFLICT for item in items): status = AnswerStatus.CONFLICTED
        else: status = AnswerStatus.ANSWERED
        return AnswerPlan(status, tuple(items), unanswered)
