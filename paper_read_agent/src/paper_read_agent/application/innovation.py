"""Scoped innovation analysis without domain-wide novelty claims."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Mapping,Protocol,Sequence
import re
from paper_read_agent.domain.evidence import Citation,EvidenceRegistry

@dataclass(frozen=True,slots=True)
class InnovationItem:
    text:str; paper_id:str; citations:tuple[Citation,...]; comparison_paper_ids:tuple[str,...]=()
@dataclass(frozen=True,slots=True)
class InnovationResult:
    author_claims:tuple[InnovationItem,...]; agent_hypotheses:tuple[InnovationItem,...]
    comparison_scope:tuple[str,...]; cannot_determine:tuple[str,...]
class InnovationGenerator(Protocol):
    def generate(self,registries:Mapping[str,EvidenceRegistry],comparison_scope:Sequence[str])->dict[str,Any]:...

class InnovationService:
    DOMAIN_CLAIM=re.compile(r"(?:领域|世界|首次|首创|state[- ]of[- ]the[- ]art|first ever)",re.I)
    def __init__(self,generator:InnovationGenerator): self.generator=generator
    def analyze(self,registries:Mapping[str,EvidenceRegistry],*,comparison_scope:Sequence[str])->InnovationResult:
        scope=tuple(dict.fromkeys(comparison_scope))
        if any(pid not in registries for pid in scope): raise ValueError("Comparison scope exceeds local registries")
        raw=self.generator.generate(registries,scope)
        def items(name):
            result=[]
            for value in raw.get(name,[]):
                text=str(value["text"]); pid=str(value["paper_id"])
                if pid not in registries: raise ValueError("Innovation item outside paper scope")
                if self.DOMAIN_CLAIM.search(text): raise ValueError("Domain-wide novelty claims are prohibited")
                citations=registries[pid].resolve(tuple(value.get("evidence_ids",())))
                if not citations: raise ValueError("Innovation claims require Evidence")
                compared=tuple(value.get("comparison_paper_ids",()))
                if any(x not in scope for x in compared): raise ValueError("Comparison exceeds declared scope")
                result.append(InnovationItem(text,pid,citations,compared))
            return tuple(result)
        cannot=tuple(raw.get("cannot_determine",()))
        if len(scope)<2 and raw.get("agent_hypotheses"):
            cannot=tuple(dict.fromkeys((*cannot,"Insufficient comparable papers in the local library")))
            hypotheses=()
        else: hypotheses=items("agent_hypotheses")
        return InnovationResult(items("author_claims"),hypotheses,scope,cannot)
