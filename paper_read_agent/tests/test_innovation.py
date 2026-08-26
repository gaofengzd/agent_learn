import pytest
from paper_read_agent.application.innovation import InnovationService
from paper_read_agent.domain.evidence import Evidence,EvidenceRegistry

def reg(pid,version="v"): return EvidenceRegistry([Evidence(f"e-{pid}-{version}","c",pid,pid,version,1,1,(),"text",.9,"native")],allowed_paper_ids=[pid])
class Gen:
    def __init__(self,data): self.data=data
    def generate(self,registries,comparison_scope): return self.data

def test_author_claims_and_agent_hypotheses_are_separate_and_scoped():
    data={"author_claims":[{"text":"作者提出模块A","paper_id":"p1","evidence_ids":["e-p1-v"]}],
          "agent_hypotheses":[{"text":"在本地库中结构不同","paper_id":"p1","evidence_ids":["e-p1-v"],"comparison_paper_ids":["p2"]}]}
    r=InnovationService(Gen(data)).analyze({"p1":reg("p1"),"p2":reg("p2")},comparison_scope=["p1","p2"])
    assert len(r.author_claims)==len(r.agent_hypotheses)==1 and r.comparison_scope==("p1","p2")

def test_no_contribution_list_and_no_comparable_paper_are_honest():
    r=InnovationService(Gen({"agent_hypotheses":[{"text":"potential","paper_id":"p1","evidence_ids":["e-p1-v"]}]})).analyze({"p1":reg("p1")},comparison_scope=["p1"])
    assert r.author_claims==() and r.agent_hypotheses==() and r.cannot_determine

@pytest.mark.parametrize("text",["这是领域首次方法","first ever architecture","state-of-the-art novelty"])
def test_domain_first_claims_are_rejected(text):
    data={"author_claims":[{"text":text,"paper_id":"p1","evidence_ids":["e-p1-v"]}]}
    with pytest.raises(ValueError,match="prohibited"): InnovationService(Gen(data)).analyze({"p1":reg("p1")},comparison_scope=["p1"])

def test_multi_version_evidence_remains_version_specific():
    registry=reg("p1","v2"); data={"author_claims":[{"text":"贡献","paper_id":"p1","evidence_ids":["e-p1-v2"]}]}
    result=InnovationService(Gen(data)).analyze({"p1":registry},comparison_scope=["p1"])
    assert "v2" in result.author_claims[0].citations[0].label
