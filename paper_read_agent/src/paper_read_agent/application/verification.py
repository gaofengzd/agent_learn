"""Post-generation claim and citation verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from paper_read_agent.application.sufficiency import AnswerPlan, SupportLevel
from paper_read_agent.domain.evidence import Evidence, EvidenceRegistry
from paper_read_agent.domain.models import AnswerStatus
from paper_read_agent.llm.glm_client import Claim, StructuredAnswer


class ClaimJudge(Protocol):
    def verify(self, claim: str, evidence: Sequence[Evidence]) -> SupportLevel: ...


@dataclass(frozen=True, slots=True)
class VerificationResult:
    answer: StructuredAnswer
    removed_claims: tuple[str, ...]
    warnings: tuple[str, ...]
    severe_failure: bool


class PostGenerationVerifier:
    def __init__(self, judge: ClaimJudge) -> None: self.judge = judge

    def verify(self, answer: StructuredAnswer, plan: AnswerPlan,
               registry: EvidenceRegistry) -> VerificationResult:
        valid: list[Claim] = []; removed: list[str] = []; warnings: list[str] = []
        evidence_by_id = {item.evidence_id: item for item in registry.evidence}
        conflicts = list(answer.conflicts)
        for claim in answer.claims:
            try: registry.resolve(claim.evidence_ids)
            except ValueError as exc:
                removed.append(claim.text); warnings.append(str(exc)); continue
            bound = [evidence_by_id[eid] for eid in claim.evidence_ids]
            if not bound:
                removed.append(claim.text); warnings.append("Claim has no evidence"); continue
            support = self.judge.verify(claim.text, bound)
            if support is SupportLevel.INSUFFICIENT:
                removed.append(claim.text); warnings.append(f"Unsupported claim removed: {claim.text}"); continue
            if support is SupportLevel.INFERENCE and claim.support != "inference":
                warnings.append(f"Claim downgraded to inference: {claim.text}")
            if support is SupportLevel.CONFLICT and claim.text not in conflicts:
                conflicts.append(claim.text); warnings.append(f"Hidden conflict exposed: {claim.text}")
            valid.append(replace(claim, support=support.value))
        severe = bool(answer.claims) and not valid
        if severe:
            status = AnswerStatus.INSUFFICIENT_EVIDENCE
            concise = "现有原文证据不足，无法可靠回答。"
            refusal = "All generated factual claims failed evidence verification"
        else:
            status = answer.answer_status
            if removed and valid: status = AnswerStatus.PARTIALLY_ANSWERED
            if conflicts: status = AnswerStatus.CONFLICTED
            concise, refusal = answer.concise_answer, answer.refusal_reason
        verified = StructuredAnswer(status, concise, tuple(valid), answer.uncertainty,
                                    tuple(conflicts), tuple(dict.fromkeys((*answer.unanswered_items,
                                        *plan.unanswered_items))), refusal)
        return VerificationResult(verified, tuple(removed), tuple(warnings), severe)
