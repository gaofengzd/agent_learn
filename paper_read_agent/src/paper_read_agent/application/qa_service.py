"""Grounded multi-turn question-answering orchestration."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from typing import Protocol, Sequence
from uuid import uuid4
from paper_read_agent.application.sufficiency import AnswerPlan
from paper_read_agent.application.verification import PostGenerationVerifier
from paper_read_agent.domain.evidence import Citation, EvidenceRegistry
from paper_read_agent.domain.models import AnswerStatus, Message, MessageRole
from paper_read_agent.llm.glm_client import GLMClient, PromptRegistry, StructuredAnswer
from paper_read_agent.persistence.repositories import SQLiteDomainRepository
from paper_read_agent.retrieval.hybrid import HybridRetriever
from paper_read_agent.retrieval.query_planner import QueryPlanner, RetrievalPlan
from paper_read_agent.retrieval.reranker import CandidateReranker, RerankResult
from paper_read_agent.application.resource_limits import ResourceGuard

@dataclass(frozen=True, slots=True)
class PreparedEvidence:
    registry: EvidenceRegistry
    answer_plan: AnswerPlan

class EvidencePreparer(Protocol):
    def prepare(self, plan: RetrievalPlan, reranked: RerankResult) -> PreparedEvidence: ...

@dataclass(frozen=True, slots=True)
class QAResult:
    answer: StructuredAnswer
    citations: tuple[Citation, ...]
    actual_paper_ids: tuple[str, ...]
    retrieval_degraded: bool

class QuestionAnsweringService:
    def __init__(self, repository: SQLiteDomainRepository, planner: QueryPlanner,
                 retriever: HybridRetriever, reranker: CandidateReranker,
                 preparer: EvidencePreparer, glm: GLMClient,
                 verifier: PostGenerationVerifier, resource_guard: ResourceGuard | None = None) -> None:
        self.repository, self.planner, self.retriever = repository, planner, retriever
        self.reranker, self.preparer, self.glm, self.verifier = reranker, preparer, glm, verifier
        self.resource_guard = resource_guard

    def answer(self, question: str, *, conversation_id: str, scope_mode: str,
               paper_ids: Sequence[str], conversation_questions: Sequence[str] = ()) -> QAResult:
        if self.resource_guard is not None: question = self.resource_guard.validate_question(question)
        plan = self.planner.plan(question, scope_mode=scope_mode,
            selected_paper_ids=paper_ids if scope_mode == "selected" else (),
            library_paper_ids=paper_ids if scope_mode == "library" else (),
            conversation_questions=conversation_questions)
        if plan.needs_clarification:
            return self._persist(conversation_id, question, plan,
                self._refusal(AnswerStatus.OUT_OF_SCOPE, plan.clarification_reason or "Need clarification"), (), False)
        recall = self.retriever.retrieve(plan)
        reranked = self.reranker.rerank(plan, recall.candidates)
        prepared = self.preparer.prepare(plan, reranked)
        if prepared.answer_plan.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
            answer = self._refusal(AnswerStatus.INSUFFICIENT_EVIDENCE,
                                   "The supplied papers do not contain sufficient evidence")
        else:
            evidence_json = json.dumps([asdict(item) for item in prepared.registry.evidence], ensure_ascii=False)
            generated = self.glm.generate(PromptRegistry.messages(
                f"Question: {plan.resolved_question}\nAnswer plan: {prepared.answer_plan}", evidence_json))
            answer = self.verifier.verify(generated, prepared.answer_plan, prepared.registry).answer
        ids = tuple(dict.fromkeys(eid for claim in answer.claims for eid in claim.evidence_ids))
        return self._persist(conversation_id, question, plan, answer, prepared.registry.resolve(ids),
                             recall.degraded or reranked.degraded)

    def _persist(self, conversation_id: str, question: str, plan: RetrievalPlan,
                 answer: StructuredAnswer, citations: Sequence[Citation], degraded: bool) -> QAResult:
        scope = {"paper_ids": list(plan.paper_ids)}
        user = Message(str(uuid4()), conversation_id, MessageRole.USER, question, retrieval_scope=scope)
        assistant = Message(str(uuid4()), conversation_id, MessageRole.ASSISTANT, answer.concise_answer,
            structured_payload=asdict(answer), retrieval_scope=scope,
            evidence_ids=tuple(x.evidence_id for x in citations), answer_status=answer.answer_status)
        self.repository.create_message_pair(user, assistant)
        return QAResult(answer, tuple(citations), plan.paper_ids, degraded)

    @staticmethod
    def _refusal(status: AnswerStatus, reason: str) -> StructuredAnswer:
        return StructuredAnswer(status, "根据当前论文原文，我不知道。", (), (reason,), (), (), reason)
