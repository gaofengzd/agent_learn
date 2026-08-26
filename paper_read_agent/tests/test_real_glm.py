import os
import pytest
from paper_read_agent.llm.glm_client import GLMClient,GLMHTTPTransport,PromptRegistry

@pytest.mark.real_glm
@pytest.mark.skipif(not os.getenv("ZHIPUAI_API_KEY"),reason="set ZHIPUAI_API_KEY to opt in")
def test_real_glm_returns_the_required_grounded_schema():
    client=GLMClient(GLMHTTPTransport(os.environ["ZHIPUAI_API_KEY"]),max_retries=0)
    messages=PromptRegistry.messages("回答证据中的字母；引用 e1。",'{"evidence_id":"e1","text":"字母是 A"}')
    answer=client.generate(messages,max_tokens=512)
    assert answer.concise_answer and all("e1" in claim.evidence_ids for claim in answer.claims)
