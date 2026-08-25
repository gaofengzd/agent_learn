import pytest

from paper_read_agent.retrieval.query_planner import QueryIntent, QueryPlanner


def test_simple_question_preserves_original_and_selected_scope() -> None:
    plan = QueryPlanner().plan("方法是什么？", scope_mode="selected", selected_paper_ids=["p1"])
    assert plan.original_question == "方法是什么？"
    assert plan.paper_ids == ("p1",)
    assert plan.intent is QueryIntent.METHOD
    assert plan.needs_clarification is False
    assert len(plan.subqueries) <= 4


def test_compound_bilingual_question_produces_bounded_subqueries() -> None:
    plan = QueryPlanner().plan(
        "比较 BGE 和 BM25；and also 说明创新点；总结结果；额外问题；第五项",
        scope_mode="library", library_paper_ids=["p1", "p2"],
    )
    assert plan.intent is QueryIntent.COMPARISON
    assert 2 <= len(plan.subqueries) <= 4
    assert any("BGE" in item and "BM25" in item for item in plan.subqueries)
    assert plan.paper_ids == ("p1", "p2")


def test_related_work_enables_references_only_for_that_intent() -> None:
    related = QueryPlanner().plan("相关工作引用了谁？", scope_mode="library",
                                  library_paper_ids=["p1"])
    normal = QueryPlanner().plan("方法是什么？", scope_mode="library", library_paper_ids=["p1"])
    assert related.include_references is True
    assert related.content_type_preferences[0] == "reference"
    assert normal.include_references is False


def test_ambiguous_pronoun_requests_clarification_without_guessing() -> None:
    plan = QueryPlanner().plan("这篇论文的创新是什么？", scope_mode="library",
                               library_paper_ids=["p1", "p2"])
    assert plan.needs_clarification is True
    assert plan.subqueries == ()
    assert plan.paper_ids == ("p1", "p2")


def test_single_paper_pronoun_is_resolved_and_followup_uses_history() -> None:
    single = QueryPlanner().plan("这篇论文的方法？", scope_mode="selected",
                                 selected_paper_ids=["p1"])
    followup = QueryPlanner().plan("为什么？", scope_mode="selected",
                                   selected_paper_ids=["p1"],
                                   conversation_questions=["作者使用了 BGE"])
    assert "paper_id=p1" in single.resolved_question
    assert "作者使用了 BGE" in followup.resolved_question
    assert followup.original_question == "为什么？"


def test_selected_scope_without_selection_requires_clarification() -> None:
    plan = QueryPlanner().plan("总结论文", scope_mode="selected")
    assert plan.needs_clarification is True
    assert plan.paper_ids == ()


def test_invalid_scope_and_empty_question_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        QueryPlanner().plan(" ", scope_mode="library")
    with pytest.raises(ValueError, match="Scope mode"):
        QueryPlanner().plan("question", scope_mode="all")
