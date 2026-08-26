import pytest
from paper_read_agent.application.sessions import SessionService
from paper_read_agent.domain.models import Message,MessageRole,Paper
from paper_read_agent.persistence import SQLiteDatabase,SQLiteDomainRepository
from paper_read_agent.retrieval.query_planner import QueryPlanner

def setup_service(tmp_path):
    db=SQLiteDatabase(tmp_path/"db");db.initialize();r=SQLiteDomainRepository(db)
    r.create_paper(Paper("a","ha","A",file_path="a"));r.create_paper(Paper("b","hb","B",file_path="b"))
    return SessionService(r),r

def test_create_restore_messages_and_restart(tmp_path):
    service,r=setup_service(tmp_path);conv=service.create("chat","selected",["a"])
    r.create_message(Message("m",conv.conversation_id,MessageRole.USER,"q"))
    restored=SessionService(r).restore(conv.conversation_id)
    assert restored.conversation.selected_paper_ids==("a",) and restored.messages[0].content=="q"

def test_scope_switch_a_to_b_and_selected_to_library_has_history(tmp_path):
    service,_=setup_service(tmp_path);conv=service.create("chat","selected",["a"])
    service.change_scope(conv.conversation_id,"selected",["b"]);service.change_scope(conv.conversation_id,"library",["a","b"])
    restored=service.restore(conv.conversation_id)
    assert restored.conversation.selected_paper_ids==("a","b") and len(restored.scope_history)==3

def test_cross_scope_ambiguous_pronoun_requires_clarification(tmp_path):
    service,_=setup_service(tmp_path);conv=service.create("chat","library",["a","b"])
    plan=QueryPlanner().plan("这篇论文的方法",scope_mode=conv.scope_mode,library_paper_ids=conv.selected_paper_ids)
    assert plan.needs_clarification

def test_delete_cascades_messages_and_history(tmp_path):
    service,r=setup_service(tmp_path);conv=service.create("chat","selected",["a"]);service.delete(conv.conversation_id)
    with pytest.raises(KeyError): service.restore(conv.conversation_id)
