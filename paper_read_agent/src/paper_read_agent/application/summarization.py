"""Hierarchical evidence-bound paper summaries."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence
from paper_read_agent.domain.evidence import Citation, EvidenceRegistry

class SummaryLevel(StrEnum): QUICK="quick"; STANDARD="standard"; DETAILED="detailed"
class SummaryGenerator(Protocol):
    def generate(self, level: SummaryLevel, registries: Mapping[str, EvidenceRegistry]) -> dict[str, Any]: ...
@dataclass(frozen=True, slots=True)
class SummarySection:
    name: str; text: str; paper_id: str; citations: tuple[Citation, ...]
@dataclass(frozen=True, slots=True)
class SummaryResult:
    level: SummaryLevel; sections: tuple[SummarySection, ...]; comparison: tuple[str, ...]
    conflicts: tuple[str, ...]; incomparable: tuple[str, ...]; warnings: tuple[str, ...]

class SummarizationService:
    REQUIRED={SummaryLevel.QUICK:("overview",), SummaryLevel.STANDARD:("abstract","method","results","limitations"),
              SummaryLevel.DETAILED:("abstract","introduction","method","experiments","conclusion","limitations")}
    def __init__(self,generator:SummaryGenerator): self.generator=generator
    def summarize(self,level:SummaryLevel,registries:Mapping[str,EvidenceRegistry],
                  *,quality_warnings:Mapping[str,Sequence[str]]|None=None)->SummaryResult:
        if not registries: raise ValueError("At least one paper is required")
        raw=self.generator.generate(level,registries); sections=[]; present=set()
        for item in raw.get("sections",[]):
            pid=str(item["paper_id"])
            if pid not in registries: raise ValueError("Summary section is outside paper scope")
            name=str(item["name"]); present.add(name)
            sections.append(SummarySection(name,str(item["text"]),pid,
                registries[pid].resolve(tuple(item.get("evidence_ids",())))))
        missing=[name for name in self.REQUIRED[level] if name not in present]
        warnings=[f"Missing summary section: {name}" for name in missing]
        for pid, values in (quality_warnings or {}).items(): warnings.extend(f"{pid}: {x}" for x in values)
        return SummaryResult(level,tuple(sections),tuple(raw.get("comparison",())),
            tuple(raw.get("conflicts",())),tuple(raw.get("incomparable",())),tuple(warnings))
