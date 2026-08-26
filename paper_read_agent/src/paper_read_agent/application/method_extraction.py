"""Evidence-bound structured method extraction."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
from paper_read_agent.domain.evidence import Citation, EvidenceRegistry

class MethodRole(StrEnum): CORE="core"; FOUNDATION="foundation"; BASELINE="baseline"; ABLATION="ablation"; AUXILIARY="auxiliary"
class SourceStatus(StrEnum): EXPLICIT="explicit"; SYNTHESIZED="synthesized"; NOT_STATED="not_stated"
FIELDS=("objective","inputs_outputs","assumptions","modules","workflow","training_objective","data","parameters","evaluation")
@dataclass(frozen=True, slots=True)
class MethodField:
    name:str; value:str|None; source_status:SourceStatus; citations:tuple[Citation,...]
@dataclass(frozen=True, slots=True)
class ExtractedMethod:
    name:str; role:MethodRole; fields:tuple[MethodField,...]
class MethodGenerator(Protocol):
    def generate(self,registry:EvidenceRegistry)->dict[str,Any]: ...

class MethodExtractionService:
    def __init__(self,generator:MethodGenerator): self.generator=generator
    def extract(self,registry:EvidenceRegistry)->tuple[ExtractedMethod,...]:
        raw=self.generator.generate(registry); methods=[]
        for item in raw.get("methods",[]):
            supplied={field["name"]:field for field in item.get("fields",[])}; fields=[]
            for name in FIELDS:
                value=supplied.get(name)
                if value is None:
                    fields.append(MethodField(name,None,SourceStatus.NOT_STATED,())); continue
                status=SourceStatus(value["source_status"]); text=value.get("value")
                citations=registry.resolve(tuple(value.get("evidence_ids",())))
                if status is SourceStatus.NOT_STATED:
                    text=None; citations=()
                elif not citations: raise ValueError("Stated method fields require Evidence")
                fields.append(MethodField(name,None if text is None else str(text),status,citations))
            methods.append(ExtractedMethod(str(item["name"]),MethodRole(item["role"]),tuple(fields)))
        return tuple(methods)
