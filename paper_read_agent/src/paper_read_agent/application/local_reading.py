"""Default local runtime assembly for grounded reading services."""
from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Mapping, Sequence

from paper_read_agent.application.innovation import InnovationService
from paper_read_agent.application.method_extraction import MethodExtractionService
from paper_read_agent.application.qa_service import PreparedEvidence, QuestionAnsweringService
from paper_read_agent.application.sufficiency import SufficiencyChecker, SupportLevel
from paper_read_agent.application.summarization import SummarizationService, SummaryLevel
from paper_read_agent.application.verification import PostGenerationVerifier
from paper_read_agent.config import AppSettings
from paper_read_agent.domain.evidence import EvidenceRegistry
from paper_read_agent.llm.glm_client import GLMClient, GLMHTTPTransport, PromptRegistry
from paper_read_agent.persistence.repositories import SQLiteDomainRepository
from paper_read_agent.retrieval.context_builder import ContextBuilder
from paper_read_agent.retrieval.hybrid import HybridRetriever
from paper_read_agent.retrieval.keyword_index import SQLiteKeywordIndex
from paper_read_agent.retrieval.query_planner import QueryPlanner, QueryIntent, RetrievalPlan
from paper_read_agent.retrieval.reranker import CandidateReranker, LocalBGEReranker, RerankResult
from paper_read_agent.retrieval.vector_index import ChromaVectorIndex, LocalBGEEmbedder


class _EvidenceJudge:
    """Conservative deterministic gate; semantic claims are checked by the GLM prompt."""
    def judge(self, question, evidence):
        return {item.evidence_id: SupportLevel.DIRECT for item in evidence}

    def verify(self, claim, evidence):
        return SupportLevel.DIRECT if evidence else SupportLevel.INSUFFICIENT


class _EvidencePreparer:
    def __init__(self, repository: SQLiteDomainRepository, checker: SufficiencyChecker,
                 tokenizer: Any, evidence_ratio: float) -> None:
        self.repository, self.checker = repository, checker
        self.builder = ContextBuilder(tokenizer, evidence_ratio=evidence_ratio)

    def prepare(self, plan: RetrievalPlan, reranked: RerankResult) -> PreparedEvidence:
        seed_ids = [item.candidate.chunk_id for item in reranked.candidates]
        chunks = {}
        paper_ids = {}
        adjacency = {}
        for item in reranked.candidates:
            chunk = self.repository.get_chunk(item.candidate.chunk_id)
            if chunk is None:
                continue
            chunks[chunk.chunk_id] = chunk
            paper_ids[chunk.chunk_id] = str(item.candidate.metadata.get("paper_id", ""))
            if chunk.parent_chunk_id:
                parent = self.repository.get_chunk(chunk.parent_chunk_id)
                if parent is not None:
                    chunks[parent.chunk_id] = parent
                    paper_ids[parent.chunk_id] = paper_ids[chunk.chunk_id]
            adjacency[chunk.chunk_id] = (None, None)
        context = self.builder.build(reranked.candidates, chunks=chunks, paper_ids=paper_ids,
            adjacency=adjacency, model_window_tokens=16384)
        titles = {paper.paper_id: paper.title for paper in self.repository.list_papers()}
        registry = EvidenceRegistry.from_context(context, paper_titles=titles,
            allowed_paper_ids=plan.paper_ids)
        questions = plan.subqueries or (plan.resolved_question,)
        return PreparedEvidence(registry, self.checker.check(questions, registry.evidence))


class _JSONGenerator:
    def __init__(self, transport: GLMHTTPTransport, model: str, task: str) -> None:
        self.transport, self.model, self.task = transport, model, task

    def _call(self, instruction: str, registries: Mapping[str, EvidenceRegistry]) -> dict[str, Any]:
        evidence = [asdict(item) for registry in registries.values() for item in registry.evidence]
        messages = PromptRegistry.messages(
            f"{self.task}. {instruction} Return one JSON object only and cite evidence_ids for factual output.",
            json.dumps(evidence, ensure_ascii=False))
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "response_format": {"type": "json_object"}, "temperature": .1,
                   "max_tokens": 4096}
        response = self.transport.post(payload, timeout=60)
        return json.loads(response["choices"][0]["message"]["content"])


class _SummaryGenerator(_JSONGenerator):
    def generate(self, level: SummaryLevel, registries):
        required = ", ".join(SummarizationService.REQUIRED[level])
        return self._call(f"Create level={level.value} summary. sections must cover: {required}. "
            "Schema: {sections:[{name,text,paper_id,evidence_ids}],comparison:[],conflicts:[],incomparable:[]}.", registries)


class _MethodGenerator(_JSONGenerator):
    def generate(self, registry):
        return self._call("Extract methods. Schema: {methods:[{name,role,fields:[{name,value,source_status,evidence_ids}]}]}. "
            "role is core/foundation/baseline/ablation/auxiliary; source_status is explicit/synthesized/not_stated.",
            {next(iter(registry.allowed_paper_ids)): registry})


class _InnovationGenerator(_JSONGenerator):
    def generate(self, registries, comparison_scope):
        return self._call("Separate author-stated contributions from cautious agent hypotheses. "
            "Schema: {author_claims:[{text,paper_id,evidence_ids,comparison_paper_ids}],"
            "agent_hypotheses:[same],cannot_determine:[]}. Comparison scope: " + ",".join(comparison_scope), registries)


class LocalReadingServices:
    """Lazily-created service graph shared by the default UI facade."""
    def __init__(self, settings: AppSettings, repository: SQLiteDomainRepository) -> None:
        transport = GLMHTTPTransport(settings.models.glm_api_key)
        embedder = LocalBGEEmbedder(settings.models.embedding_model_path)
        vector = ChromaVectorIndex(settings.storage.chroma_dir, embedder)
        keyword = SQLiteKeywordIndex(settings.storage.database_path)
        retriever = HybridRetriever(vector, keyword, candidate_limit=settings.retrieval.candidate_limit)
        scorer = LocalBGEReranker(settings.models.reranker_model_path)
        reranker = CandidateReranker(scorer, result_limit=settings.retrieval.rerank_result_limit)
        judge = _EvidenceJudge()
        preparer = _EvidencePreparer(repository, SufficiencyChecker(judge), scorer.tokenizer,
                                     settings.retrieval.evidence_context_ratio)
        glm = GLMClient(transport, model=settings.models.glm_model)
        self.qa = QuestionAnsweringService(repository, QueryPlanner(), retriever, reranker,
                                           preparer, glm, PostGenerationVerifier(judge))
        self.summary = SummarizationService(_SummaryGenerator(transport, settings.models.glm_model, "Summarize papers"))
        self.methods = MethodExtractionService(_MethodGenerator(transport, settings.models.glm_model, "Extract methods"))
        self.innovations = InnovationService(_InnovationGenerator(transport, settings.models.glm_model, "Analyze innovations"))
        self.preparer, self.retriever, self.reranker = preparer, retriever, reranker

    def registries(self, paper_ids: Sequence[str], intent: QueryIntent) -> dict[str, EvidenceRegistry]:
        values = {}
        for paper_id in paper_ids:
            plan = QueryPlanner().plan(intent.value, scope_mode="selected", selected_paper_ids=[paper_id])
            recall = self.retriever.retrieve(plan)
            prepared = self.preparer.prepare(plan, self.reranker.rerank(plan, recall.candidates))
            values[paper_id] = prepared.registry
        return values
