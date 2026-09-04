import json
import traceback
from urllib import error
import pytest

from paper_read_agent.domain.models import AnswerStatus
from paper_read_agent.exceptions import ModelInvocationError
from paper_read_agent.llm.glm_client import GLMClient, GLMHTTPTransport, PromptRegistry


def valid(): return {"answer_status": "answered", "concise_answer": "yes",
    "claims": [{"text": "fact", "evidence_ids": ["ev_1"], "support": "direct"}],
    "uncertainty": [], "conflicts": [], "unanswered_items": [], "refusal_reason": None}


class FakeTransport:
    def __init__(self, outputs): self.outputs=list(outputs); self.calls=[]
    def post(self, payload, *, timeout):
        self.calls.append((payload, timeout)); value=self.outputs.pop(0)
        if isinstance(value, Exception): raise value
        return {"choices": [{"message": {"content": value}}]}


def test_valid_json_is_parsed_and_prompt_is_versioned():
    transport=FakeTransport([json.dumps(valid())]); client=GLMClient(transport, max_retries=0)
    result=client.generate(PromptRegistry.messages("answer", "[]"))
    assert result.answer_status is AnswerStatus.ANSWERED and result.claims[0].evidence_ids == ("ev_1",)
    assert PromptRegistry.VERSION and transport.calls[0][0]["model"] == "glm-4.7"
    assert transport.calls[0][0]["response_format"] == {"type": "json_object"}
    system_prompt = transport.calls[0][0]["messages"][0]["content"]
    assert all(field in system_prompt for field in valid())
    assert "Do not wrap the object in another field" in system_prompt


def test_non_qa_evidence_prompt_does_not_force_qa_schema():
    messages = PromptRegistry.evidence_messages("extract methods", "[]")

    assert "Use only supplied Evidence" in messages[0]["content"]
    assert "answer_status" not in messages[0]["content"]
    assert "extract methods" in messages[1]["content"]


@pytest.mark.parametrize("mutate", [
    lambda x: x.pop("claims"), lambda x: x.update(answer_status="invalid"),
    lambda x: x["claims"][0].update(support="maybe")])
def test_missing_and_invalid_fields_are_rejected(mutate):
    data=valid(); mutate(data)
    with pytest.raises(ModelInvocationError): GLMClient(FakeTransport([json.dumps(data)]), max_retries=0).generate([])


def test_non_json_empty_timeout_and_rate_limit_obey_retry_limit():
    for bad in ["not json", "", ModelInvocationError("GLM HTTP 429"), TimeoutError("slow")]:
        transport=FakeTransport([bad, bad])
        with pytest.raises((ModelInvocationError, TimeoutError)):
            GLMClient(transport, max_retries=1).generate([])
        assert len(transport.calls) == 2


def test_retry_can_recover_without_exposing_key():
    transport=FakeTransport([ModelInvocationError("GLM HTTP 500"), json.dumps(valid())])
    assert GLMClient(transport, max_retries=1).generate([]).concise_answer == "yes"


def test_http_failure_suppresses_credential_bearing_urllib_chain(monkeypatch):
    secret = "test-key-must-never-appear"

    def fail(*args, **kwargs):
        raise error.URLError(f"Authorization: Bearer {secret}")

    monkeypatch.setattr("paper_read_agent.llm.glm_client.request.urlopen", fail)
    with pytest.raises(ModelInvocationError) as caught:
        GLMHTTPTransport(secret).post({"model": "glm-4.7"}, timeout=1)

    rendered = "".join(traceback.format_exception(caught.value))
    assert secret not in rendered
    assert "Authorization" not in rendered
    assert caught.value.__suppress_context__ is True


def test_remote_disconnect_is_normalized_for_caller_retry(monkeypatch):
    from http.client import RemoteDisconnected

    def fail(*args, **kwargs):
        raise RemoteDisconnected("remote closed the connection")

    monkeypatch.setattr("paper_read_agent.llm.glm_client.request.urlopen", fail)

    with pytest.raises(ModelInvocationError, match="RemoteDisconnected") as caught:
        GLMHTTPTransport("secret").post({"model": "glm-4.7"}, timeout=1)

    assert caught.value.__suppress_context__ is True
