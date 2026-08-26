import pytest
from paper_read_agent.application.summarization import SummarizationService,SummaryLevel
from paper_read_agent.domain.evidence import Evidence,EvidenceRegistry

def registry(pid): return EvidenceRegistry([Evidence(f"e-{pid}","c",pid,pid,"v",1,1,(),"text",.9,"native")],allowed_paper_ids=[pid])
class Generator:
    def __init__(self,data): self.data=data
    def generate(self,level,registries): return self.data

@pytest.mark.parametrize("level,names",[(SummaryLevel.QUICK,["overview"]),(SummaryLevel.STANDARD,["abstract","method","results","limitations"]),(SummaryLevel.DETAILED,["abstract","introduction","method","experiments","conclusion","limitations"])])
def test_three_levels_have_required_density_and_citations(level,names):
    data={"sections":[{"name":x,"text":x,"paper_id":"p","evidence_ids":["e-p"]} for x in names]}
    result=SummarizationService(Generator(data)).summarize(level,{"p":registry("p")})
    assert len(result.sections)==len(names) and not result.warnings and result.sections[0].citations

def test_multi_paper_comparison_conflicts_and_incomparable_are_preserved():
    data={"sections":[{"name":"overview","text":"a","paper_id":"a","evidence_ids":["e-a"]}],
          "comparison":["different methods"],"conflicts":["result conflict"],"incomparable":["different datasets"]}
    r=SummarizationService(Generator(data)).summarize(SummaryLevel.QUICK,{"a":registry("a"),"b":registry("b")})
    assert r.comparison and r.conflicts and r.incomparable

def test_partial_quality_and_missing_fields_produce_warnings():
    r=SummarizationService(Generator({"sections":[]})).summarize(SummaryLevel.DETAILED,{"p":registry("p")},quality_warnings={"p":["missing page"]})
    assert any("Missing summary section" in x for x in r.warnings) and any("missing page" in x for x in r.warnings)

def test_scope_escape_and_invalid_citation_fail():
    for data in [{"sections":[{"name":"overview","text":"x","paper_id":"other"}]},
                 {"sections":[{"name":"overview","text":"x","paper_id":"p","evidence_ids":["fake"]}]}]:
        with pytest.raises(ValueError): SummarizationService(Generator(data)).summarize(SummaryLevel.QUICK,{"p":registry("p")})
