"""Deterministic, scope-safe retrieval planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Sequence


class QueryIntent(StrEnum):
    FACTUAL = "factual"
    SUMMARY = "summary"
    METHOD = "method"
    INNOVATION = "innovation"
    COMPARISON = "comparison"
    RELATED_WORK = "related_work"


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    original_question: str
    resolved_question: str
    intent: QueryIntent
    subqueries: tuple[str, ...]
    paper_ids: tuple[str, ...]
    content_type_preferences: tuple[str, ...]
    include_references: bool
    needs_clarification: bool = False
    clarification_reason: str | None = None


class QueryPlanner:
    _PRONOUN_PATTERN = re.compile(r"(?:这|那|该)(?:篇|个)?(?:论文|文章|文献)|它(?:的|们)?|this paper", re.I)

    def plan(
        self,
        question: str,
        *,
        scope_mode: str,
        selected_paper_ids: Sequence[str] = (),
        library_paper_ids: Sequence[str] = (),
        conversation_questions: Sequence[str] = (),
    ) -> RetrievalPlan:
        original = question.strip()
        if not original:
            raise ValueError("Question must not be empty")
        if scope_mode not in {"selected", "library"}:
            raise ValueError("Scope mode must be selected or library")
        paper_ids = tuple(dict.fromkeys(
            selected_paper_ids if scope_mode == "selected" else library_paper_ids
        ))
        ambiguous = bool(self._PRONOUN_PATTERN.search(original)) and len(paper_ids) != 1
        missing_selected_scope = scope_mode == "selected" and not paper_ids
        if ambiguous or missing_selected_scope:
            reason = (
                "The referenced paper is ambiguous within the current scope"
                if ambiguous else "No paper is selected for selected-paper scope"
            )
            return RetrievalPlan(
                original, original, self._intent(original), (), paper_ids, (), False,
                needs_clarification=True, clarification_reason=reason,
            )

        resolved = original
        if self._PRONOUN_PATTERN.search(original) and len(paper_ids) == 1:
            resolved = f"{original} [paper_id={paper_ids[0]}]"
        elif self._looks_like_follow_up(original) and conversation_questions:
            resolved = f"{conversation_questions[-1]}；追问：{original}"
        intent = self._intent(resolved)
        include_references = intent is QueryIntent.RELATED_WORK
        preferences = self._preferences(intent)
        subqueries = self._subqueries(resolved)
        return RetrievalPlan(
            original, resolved, intent, subqueries, paper_ids, preferences,
            include_references,
        )

    @staticmethod
    def _intent(question: str) -> QueryIntent:
        value = question.casefold()
        rules = (
            (QueryIntent.RELATED_WORK, ("相关工作", "参考文献", "related work", "citation")),
            (QueryIntent.COMPARISON, ("比较", "区别", "异同", "compare", "versus", " vs ")),
            (QueryIntent.INNOVATION, ("创新", "贡献", "novel", "contribution")),
            (QueryIntent.METHOD, ("方法", "模型", "算法", "method", "architecture", "algorithm")),
            (QueryIntent.SUMMARY, ("总结", "概括", "摘要", "summary", "summarize")),
        )
        return next((intent for intent, terms in rules if any(term in value for term in terms)),
                    QueryIntent.FACTUAL)

    @staticmethod
    def _preferences(intent: QueryIntent) -> tuple[str, ...]:
        return {
            QueryIntent.METHOD: ("text", "formula", "table", "caption"),
            QueryIntent.INNOVATION: ("text", "heading"),
            QueryIntent.COMPARISON: ("text", "table"),
            QueryIntent.RELATED_WORK: ("reference", "text"),
            QueryIntent.SUMMARY: ("heading", "text"),
            QueryIntent.FACTUAL: ("text", "table", "formula"),
        }[intent]

    @staticmethod
    def _subqueries(question: str) -> tuple[str, ...]:
        parts = [
            part.strip(" ，,。；;？?")
            for part in re.split(r"[；;？?]|(?:并且|以及|同时|and also|and then)", question, flags=re.I)
            if part.strip(" ，,。；;？?")
        ]
        values = list(dict.fromkeys([question, *parts]))
        return tuple(values[:4])

    @staticmethod
    def _looks_like_follow_up(question: str) -> bool:
        value = question.casefold().lstrip()
        return value.startswith(("为什么", "怎么", "如何", "那", "那么", "还有",
                                 "why", "how", "what about"))
