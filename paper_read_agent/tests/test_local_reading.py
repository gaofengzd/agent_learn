from types import SimpleNamespace
import json

import pytest

from paper_read_agent.application.local_reading import (
    _JSONGenerator, _LimitedReranker, _SummaryGenerator,
)
from paper_read_agent.application.summarization import SummaryLevel
from paper_read_agent.exceptions import ModelInvocationError


class RecordingReranker:
    def __init__(self) -> None:
        self.candidates = None

    def rerank(self, plan, candidates):
        self.candidates = candidates
        return tuple(candidates)


def test_limited_reranker_caps_input_independently_of_result_limit() -> None:
    delegate = RecordingReranker()
    reranker = _LimitedReranker(delegate, input_limit=20)
    candidates = tuple(range(50))

    result = reranker.rerank(SimpleNamespace(), candidates)

    assert delegate.candidates == candidates[:20]
    assert result == candidates[:20]


def test_limited_reranker_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _LimitedReranker(RecordingReranker(), input_limit=0)


class FakeTransport:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = 0
        self.timeouts = []

    def post(self, payload, *, timeout):
        self.calls += 1
        self.timeouts.append(timeout)
        return {"choices": [{"message": {"content": next(self.contents)}}]}


def test_json_generator_retries_empty_and_invalid_responses() -> None:
    transport = FakeTransport(["", "not-json", json.dumps({"sections": []})])
    generator = _JSONGenerator(transport, "glm-4.7", "summarize")

    assert generator._call("brief", {}) == {"sections": []}
    assert transport.calls == 3
    assert transport.timeouts == [None, None, None]


def test_json_generator_reports_retry_exhaustion() -> None:
    generator = _JSONGenerator(FakeTransport(["", ""]), "glm-4.7", "summarize",
                               max_retries=1)

    with pytest.raises(ModelInvocationError, match="after 2 attempts"):
        generator._call("standard", {})


def test_json_generator_empty_response_reports_safe_diagnostics() -> None:
    class DiagnosticTransport:
        def post(self, payload, *, timeout):
            return {"choices": [{"message": {"content": ""},
                                 "finish_reason": "length"}],
                    "usage": {"completion_tokens": 0}}

    with pytest.raises(ModelInvocationError, match="finish_reason='length'"):
        _JSONGenerator(DiagnosticTransport(), "glm-4.7", "summary",
                       max_retries=0)._call("detailed", {})


def test_detailed_summary_is_generated_one_section_at_a_time() -> None:
    names = ("abstract", "introduction", "method", "experiments",
             "conclusion", "limitations")
    outputs = [
        json.dumps({"sections": [{"name": name, "text": name, "paper_id": "p",
                                  "evidence_ids": []}],
                    "comparison": [], "conflicts": [], "incomparable": []})
        for name in names
    ]
    transport = FakeTransport(outputs)

    result = _SummaryGenerator(transport, "glm-4.7", "summary").generate(
        SummaryLevel.DETAILED, {}
    )

    assert [item["name"] for item in result["sections"]] == list(names)
    assert transport.calls == len(names)
