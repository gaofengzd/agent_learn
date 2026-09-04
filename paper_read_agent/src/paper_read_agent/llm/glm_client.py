"""GLM-4.7 client boundary, prompt registry, and strict structured output."""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time
from typing import Any, Protocol, Sequence
from urllib import error, request

from paper_read_agent.domain.models import AnswerStatus
from paper_read_agent.exceptions import ModelInvocationError


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    evidence_ids: tuple[str, ...]
    support: str


@dataclass(frozen=True, slots=True)
class StructuredAnswer:
    answer_status: AnswerStatus
    concise_answer: str
    claims: tuple[Claim, ...]
    uncertainty: tuple[str, ...]
    conflicts: tuple[str, ...]
    unanswered_items: tuple[str, ...]
    refusal_reason: str | None


class PromptRegistry:
    VERSION = "grounded-answer-v1"
    EVIDENCE_SYSTEM = (
        "You are an academic paper reader. Use only supplied Evidence. "
        "Cite only provided evidence_id values. Never invent facts, pages, or citations. "
        "If evidence is insufficient, say so explicitly and leave the item unanswered."
    )
    TRUSTED_SYSTEM = (
        EVIDENCE_SYSTEM + " "
        "Return exactly one JSON object with all of these top-level fields: "
        "answer_status, concise_answer, claims, uncertainty, conflicts, "
        "unanswered_items, refusal_reason. "
        "answer_status must be answered, partially_answered, conflicted, "
        "insufficient_evidence, document_quality_failure, or out_of_scope. "
        "claims must be an array of objects with text, evidence_ids, and support; "
        "support must be direct, inference, conflict, or unsupported. "
        "uncertainty, conflicts, and unanswered_items must be arrays of strings. "
        "refusal_reason must be a string or null. Do not wrap the object in another field."
    )

    @classmethod
    def messages(cls, task_prompt: str, evidence_json: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": cls.TRUSTED_SYSTEM},
                {"role": "user", "content": f"{task_prompt}\nEvidence:\n{evidence_json}"}]

    @classmethod
    def evidence_messages(cls, task_prompt: str, evidence_json: str) -> list[dict[str, str]]:
        """Evidence-only guardrails for non-QA structured schemas."""
        return [{"role": "system", "content": cls.EVIDENCE_SYSTEM},
                {"role": "user", "content": f"{task_prompt}\nEvidence:\n{evidence_json}"}]


class Transport(Protocol):
    def post(self, payload: dict[str, Any], *, timeout: float | None) -> dict[str, Any]: ...


class GLMHTTPTransport:
    ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def __init__(self, api_key: str, endpoint: str | None = None) -> None:
        self.api_key, self.endpoint = api_key, endpoint or self.ENDPOINT

    def post(self, payload: dict[str, Any], *, timeout: float | None) -> dict[str, Any]:
        req = request.Request(self.endpoint, data=json.dumps(payload).encode(), method="POST",
                              headers={"Authorization": f"Bearer {self.api_key}",
                                       "Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except error.HTTPError as exc:
            # urllib exceptions retain their Request, including Authorization.
            # Do not attach that credential-bearing object to the public chain.
            raise ModelInvocationError(f"GLM HTTP {exc.code}") from None
        except (error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
            raise ModelInvocationError(f"GLM transport failure: {type(exc).__name__}") from None


class GLMClient:
    def __init__(self, transport: Transport, *, model: str = "glm-4.7", timeout: float | None = None,
                 max_retries: int = 2, max_output_tokens: int = 8192) -> None:
        self.transport, self.model, self.timeout, self.max_retries = transport, model, timeout, max_retries
        self.max_output_tokens = max_output_tokens

    def generate(self, messages: Sequence[dict[str, str]], *, max_tokens: int = 4096) -> StructuredAnswer:
        if max_tokens <= 0 or max_tokens > self.max_output_tokens:
            raise ValueError("Requested output tokens exceed configured limit")
        payload = {"model": self.model, "messages": list(messages), "stream": False,
                   "response_format": {"type": "json_object"}, "max_tokens": max_tokens,
                   "temperature": 0.1}
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.transport.post(payload, timeout=self.timeout)
                content = response["choices"][0]["message"]["content"]
                if not content: raise ModelInvocationError("GLM returned an empty response")
                return self._validate(json.loads(content))
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, TimeoutError,
                    ModelInvocationError) as exc:
                last = exc
                if attempt >= self.max_retries: break
                time.sleep(min(0.05 * (2 ** attempt), 0.2))
        raise ModelInvocationError(
            f"GLM request failed after {self.max_retries + 1} attempts: {type(last).__name__}: {last}"
        ) from last

    @staticmethod
    def _validate(value: Any) -> StructuredAnswer:
        if not isinstance(value, dict): raise ModelInvocationError("Structured response must be an object")
        required = {"answer_status", "concise_answer", "claims", "uncertainty", "conflicts",
                    "unanswered_items", "refusal_reason"}
        missing = required - value.keys()
        if missing: raise ModelInvocationError("Missing structured fields: " + ", ".join(sorted(missing)))
        try: status = AnswerStatus(value["answer_status"])
        except (ValueError, TypeError) as exc: raise ModelInvocationError("Invalid answer_status") from exc
        claims = []
        for item in value["claims"]:
            if not isinstance(item, dict) or set(("text", "evidence_ids", "support")) - item.keys():
                raise ModelInvocationError("Invalid claim structure")
            if item["support"] not in {"direct", "inference", "conflict", "unsupported"}:
                raise ModelInvocationError("Invalid claim support")
            claims.append(Claim(str(item["text"]), tuple(map(str, item["evidence_ids"])), item["support"]))
        def strings(name: str) -> tuple[str, ...]:
            if not isinstance(value[name], list): raise ModelInvocationError(f"{name} must be a list")
            return tuple(map(str, value[name]))
        return StructuredAnswer(status, str(value["concise_answer"]), tuple(claims), strings("uncertainty"),
                                strings("conflicts"), strings("unanswered_items"),
                                None if value["refusal_reason"] is None else str(value["refusal_reason"]))
