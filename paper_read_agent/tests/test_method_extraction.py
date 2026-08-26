import pytest
from paper_read_agent.application.method_extraction import MethodExtractionService,MethodRole,SourceStatus,FIELDS
from paper_read_agent.domain.evidence import Evidence,EvidenceRegistry

def registry(): return EvidenceRegistry([Evidence("e","c","p","P","v",1,2,("Method",),"text",.9,"native")],allowed_paper_ids=["p"])
class Gen:
    def __init__(self,data): self.data=data
    def generate(self,registry): return self.data

def test_single_method_cross_section_fields_and_missing_parameters():
    data={"methods":[{"name":"M","role":"core","fields":[
        {"name":"objective","value":"classify","source_status":"explicit","evidence_ids":["e"]},
        {"name":"workflow","value":"A then B","source_status":"synthesized","evidence_ids":["e"]}]}]}
    method=MethodExtractionService(Gen(data)).extract(registry())[0]
    assert method.role is MethodRole.CORE and len(method.fields)==len(FIELDS)
    assert next(x for x in method.fields if x.name=="parameters").source_status is SourceStatus.NOT_STATED

def test_multiple_roles_do_not_mix_baseline_ablation_and_auxiliary():
    data={"methods":[{"name":r,"role":r,"fields":[]} for r in ["core","foundation","baseline","ablation","auxiliary"]]}
    result=MethodExtractionService(Gen(data)).extract(registry())
    assert [x.role.value for x in result]==["core","foundation","baseline","ablation","auxiliary"]

def test_not_stated_discards_fabricated_value_and_citation():
    data={"methods":[{"name":"M","role":"core","fields":[{"name":"parameters","value":"guessed 0.1","source_status":"not_stated","evidence_ids":["e"]}]}]}
    field=next(x for x in MethodExtractionService(Gen(data)).extract(registry())[0].fields if x.name=="parameters")
    assert field.value is None and field.citations==()

def test_explicit_or_synthesized_without_evidence_is_rejected():
    data={"methods":[{"name":"M","role":"core","fields":[{"name":"objective","value":"x","source_status":"explicit","evidence_ids":[]}]}]}
    with pytest.raises(ValueError,match="require Evidence"): MethodExtractionService(Gen(data)).extract(registry())
