"""Default local runtime assembly for grounded reading services."""
from __future__ import annotations

from dataclasses import asdict
import json
import time
from typing import Any, Mapping, Sequence

from paper_read_agent.application.innovation import InnovationService
from paper_read_agent.application.method_extraction import MethodExtractionService
from paper_read_agent.application.qa_service import PreparedEvidence, QuestionAnsweringService
from paper_read_agent.application.sufficiency import SufficiencyChecker, SupportLevel
from paper_read_agent.application.summarization import SummarizationService, SummaryLevel
from paper_read_agent.application.verification import PostGenerationVerifier
from paper_read_agent.config import AppSettings
from paper_read_agent.domain.evidence import EvidenceRegistry
from paper_read_agent.exceptions import ModelInvocationError
from paper_read_agent.llm.glm_client import GLMClient, GLMHTTPTransport, PromptRegistry
from paper_read_agent.persistence.repositories import SQLiteDomainRepository
from paper_read_agent.retrieval.context_builder import ContextBuilder
from paper_read_agent.retrieval.hybrid import HybridCandidate, HybridRetriever
from paper_read_agent.retrieval.keyword_index import SQLiteKeywordIndex
from paper_read_agent.retrieval.query_planner import QueryPlanner, QueryIntent, RetrievalPlan
from paper_read_agent.retrieval.reranker import (
    CandidateReranker, LocalBGEReranker, RerankedCandidate, RerankResult,
)
from paper_read_agent.retrieval.vector_index import ChromaVectorIndex, LocalBGEEmbedder


class _EvidenceJudge:
    """Conservative deterministic gate; semantic claims are checked by the GLM prompt."""
    def judge(self, question, evidence):
        return {item.evidence_id: SupportLevel.DIRECT for item in evidence}

    def verify(self, claim, evidence):
        return SupportLevel.DIRECT if evidence else SupportLevel.INSUFFICIENT


class _LimitedReranker:
    """Bound CPU work while preserving the configured number of final results."""
    def __init__(self, delegate: CandidateReranker, input_limit: int) -> None:
        if input_limit <= 0:
            raise ValueError("Rerank input limit must be positive")
        self.delegate, self.input_limit = delegate, input_limit

    def rerank(self, plan, candidates):
        return self.delegate.rerank(plan, candidates[:self.input_limit])


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
    def __init__(self, transport: GLMHTTPTransport, model: str, task: str,
                 max_retries: int = 2, timeout: float | None = None) -> None:
        self.transport, self.model, self.task = transport, model, task
        self.max_retries = max_retries
        self.timeout = timeout
        self.cancellation_check = lambda: False

    def _call(self, instruction: str, registries: Mapping[str, EvidenceRegistry]) -> dict[str, Any]:
        evidence = [asdict(item) for registry in registries.values() for item in registry.evidence]
        messages = PromptRegistry.evidence_messages(
            f"{self.task}. {instruction} Return one JSON object only and cite evidence_ids for factual output.",
            json.dumps(evidence, ensure_ascii=False))
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "response_format": {"type": "json_object"}, "temperature": .1,
                   "max_tokens": 4096}
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.cancellation_check():
                raise RuntimeError("Analysis cancelled")
            try:
                response = self.transport.post(payload, timeout=self.timeout)
                if self.cancellation_check():
                    raise RuntimeError("Analysis cancelled")
                content = response["choices"][0]["message"]["content"]
                if not content:
                    choice = response.get("choices", [{}])[0]
                    finish_reason = choice.get("finish_reason", "unknown")
                    usage = response.get("usage", {})
                    raise ModelInvocationError(
                        "GLM returned an empty response "
                        f"(finish_reason={finish_reason!r}, usage={usage!r})"
                    )
                value = json.loads(content)
                if not isinstance(value, dict):
                    raise ModelInvocationError("Structured response must be an object")
                return value
            except (json.JSONDecodeError, KeyError, IndexError, TypeError,
                    TimeoutError, ModelInvocationError) as exc:
                last = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(0.05 * (2 ** attempt), 0.2))
        raise ModelInvocationError(
            f"GLM request failed after {self.max_retries + 1} attempts: "
            f"{type(last).__name__}: {last}"
        ) from last


class _SummaryGenerator(_JSONGenerator):
    def generate(self, level: SummaryLevel, registries):
        sections = []
        comparison, conflicts, incomparable = [], [], []
        # Generate one required section at a time. Long, monolithic detailed-summary
        # responses were observed to finish with an empty body at the GLM boundary.
        for name in SummarizationService.REQUIRED[level]:
            value = self._call(
                f"Create only the {name!r} section of a level={level.value} summary. "
                "Schema: {sections:[{name,text,paper_id,evidence_ids}],comparison:[],"
                "conflicts:[],incomparable:[]}. sections must contain exactly one item "
                f"whose name is {name!r}. If Evidence is insufficient, state that the "
                "current Evidence is insufficient; do not claim the paper omitted it.",
                registries,
            )
            sections.extend(value.get("sections", ()))
            comparison.extend(value.get("comparison", ()))
            conflicts.extend(value.get("conflicts", ()))
            incomparable.extend(value.get("incomparable", ()))
        return {"sections": sections, "comparison": list(dict.fromkeys(comparison)),
                "conflicts": list(dict.fromkeys(conflicts)),
                "incomparable": list(dict.fromkeys(incomparable))}


class _MethodGenerator(_JSONGenerator):
    def generate(self, registry):
        return self._call("Extract actual named methods or systems, not one method per field. "
            "Schema: {methods:[{name,role,fields:[{name,value,source_status,evidence_ids}]}]}. "
            "Every fields array must contain exactly these field names: objective, inputs_outputs, "
            "assumptions, modules, workflow, training_objective, data, parameters, evaluation. "
            "role is core/foundation/baseline/ablation/auxiliary; source_status is "
            "explicit/synthesized/not_stated. Use explicit or synthesized with valid evidence_ids "
            "whenever supplied Evidence supports a field; reserve not_stated for genuinely unsupported fields.",
            {next(iter(registry.allowed_paper_ids)): registry})


class _InnovationGenerator(_JSONGenerator):
    def generate(self, registries, comparison_scope):
        return self._call("Separate author-stated contributions from cautious agent hypotheses. "
            "Schema: {author_claims:[{text,paper_id,evidence_ids,comparison_paper_ids}],"
            "agent_hypotheses:[same],cannot_determine:[]}. Comparison scope: " + ",".join(comparison_scope), registries)


class LocalReadingServices:
    """Lazily-created service graph shared by the default UI facade."""

    _ANALYSIS_QUERIES = {
        QueryIntent.SUMMARY: (
            "abstract overview contributions", "introduction motivation problem",
            "method architecture workflow", "experiments datasets baselines results",
            "conclusion findings", "limitations failure cases future work",
        ),
        QueryIntent.METHOD: (
            "method objective task", "method inputs outputs", "method assumptions",
            "architecture modules components", "method workflow algorithm steps",
            "training objective loss function", "datasets data construction",
            "parameters implementation settings", "evaluation metrics experiments",
        ),
        QueryIntent.INNOVATION: (
            "author stated contributions", "novel innovation proposed method",
            "comparison baselines improvements", "limitations future work",
        ),
    }
    _SECTION_HINTS = {
        QueryIntent.SUMMARY: (
            ("abstract",), ("introduction",), ("method", "methodology"),
            ("experiment",), ("result",), ("conclusion",), ("limitation",),
        ),
        QueryIntent.METHOD: (
            ("method", "methodology"), ("2.1",), ("2.2",), ("2.3",),
            ("architecture", "framework"), ("experiment setup",), ("evaluation",),
        ),
        QueryIntent.INNOVATION: (
            ("abstract",), ("introduction",), ("conclusion",), ("limitation",),
        ),
    }
    def __init__(self, settings: AppSettings, repository: SQLiteDomainRepository) -> None:
        transport = GLMHTTPTransport(settings.models.glm_api_key)
        embedder = LocalBGEEmbedder(settings.models.embedding_model_path)
        vector = ChromaVectorIndex(settings.storage.chroma_dir, embedder)
        keyword = SQLiteKeywordIndex(settings.storage.database_path)
        retriever = HybridRetriever(vector, keyword, candidate_limit=settings.retrieval.candidate_limit)
        scorer = LocalBGEReranker(settings.models.reranker_model_path)
        reranker = _LimitedReranker(
            CandidateReranker(scorer, result_limit=settings.retrieval.rerank_result_limit),
            settings.retrieval.rerank_input_limit)
        judge = _EvidenceJudge()
        preparer = _EvidencePreparer(repository, SufficiencyChecker(judge), scorer.tokenizer,
                                     settings.retrieval.evidence_context_ratio)
        glm = GLMClient(transport, model=settings.models.glm_model,
                        timeout=settings.models.qa_timeout)
        self.qa = QuestionAnsweringService(repository, QueryPlanner(), retriever, reranker,
                                           preparer, glm, PostGenerationVerifier(judge))
        self.summary = SummarizationService(_SummaryGenerator(transport, settings.models.glm_model, "Summarize papers"))
        self.methods = MethodExtractionService(_MethodGenerator(transport, settings.models.glm_model, "Extract methods"))
        self.innovations = InnovationService(_InnovationGenerator(transport, settings.models.glm_model, "Analyze innovations"))
        self.preparer, self.retriever, self.reranker = preparer, retriever, reranker
        self._analysis_generators=(self.summary.generator,self.methods.generator,
                                  self.innovations.generator)

    def set_analysis_cancellation_check(self,check):
        for generator in self._analysis_generators:
            generator.cancellation_check=check

    def registries(self, paper_ids: Sequence[str], intent: QueryIntent) -> dict[str, EvidenceRegistry]:
        values = {}
        for paper_id in paper_ids:
            queries = self._ANALYSIS_QUERIES.get(intent, (intent.value,))
            ranked: list[RerankedCandidate] = []
            input_limit = getattr(self.reranker, "input_limit", len(queries))
            per_query = max(1, input_limit // len(queries))
            result_limit = getattr(getattr(self.reranker, "delegate", None),
                                   "result_limit", 12)
            delegate = getattr(self.reranker, "delegate", self.reranker)
            for query in queries:
                if self.summary.generator.cancellation_check():
                    raise RuntimeError("Analysis cancelled")
                plan = QueryPlanner().plan(query, scope_mode="selected",
                                           selected_paper_ids=[paper_id])
                recall = self.retriever.retrieve(plan)
                result = delegate.rerank(plan, recall.candidates[:per_query])
                ranked.extend(result.candidates)
            paper = self.preparer.repository.get_paper(paper_id)
            chunks = self.preparer.repository.list_chunks(paper.active_version_id) if (
                paper is not None and paper.active_version_id
            ) else ()
            for position, hints in enumerate(self._SECTION_HINTS.get(intent, ())):
                matches = [
                    chunk for chunk in chunks
                    if any(hint.casefold() in " > ".join(chunk.section_path).casefold()
                           for hint in hints)
                    and (chunk.quality_score is None or chunk.quality_score >= .5)
                ]
                if not matches:
                    continue
                chunk = max(matches, key=lambda item: (len(item.text), -item.page_start))
                candidate = HybridCandidate(
                    chunk.chunk_id, chunk.text, 1.0,
                    {"paper_id": paper_id, "version_id": chunk.version_id,
                     "content_type": chunk.content_type},
                    (),
                )
                ranked.append(RerankedCandidate(
                    candidate, 2.0 - position * .01, ("section:" + hints[0],)
                ))
            ranked.sort(key=lambda item: (-item.score, -item.candidate.rrf_score,
                                           item.candidate.chunk_id))
            diverse = []
            seen_facets = set()
            seen_chunks = set()
            # Preserve one best result per facet, then fill remaining slots globally.
            for item in ranked:
                facet = item.matched_subqueries[0] if item.matched_subqueries else ""
                chunk_id = item.candidate.chunk_id
                if facet and facet not in seen_facets and chunk_id not in seen_chunks:
                    diverse.append(item)
                    seen_facets.add(facet)
                    seen_chunks.add(chunk_id)
            for item in ranked:
                chunk_id = item.candidate.chunk_id
                if chunk_id not in seen_chunks:
                    diverse.append(item)
                    seen_chunks.add(chunk_id)
                if len(diverse) >= result_limit:
                    break
            plan = QueryPlanner().plan("; ".join(queries), scope_mode="selected",
                                       selected_paper_ids=[paper_id])
            prepared = self.preparer.prepare(
                plan, RerankResult(tuple(diverse[:result_limit]), False)
            )
            values[paper_id] = prepared.registry
        return values
