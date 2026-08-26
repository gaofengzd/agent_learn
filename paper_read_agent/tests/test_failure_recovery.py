import sqlite3
import pytest
from paper_read_agent.application.failure_recovery import FailureKind,FailureRecovery
from paper_read_agent.exceptions import DocumentProcessingError,ModelInvocationError,RetrievalError

@pytest.mark.parametrize("stage,error,kind",[("docling",DocumentProcessingError("x"),FailureKind.PARSING),
    ("ocr",RuntimeError("x"),FailureKind.OCR),("embedding",RuntimeError("x"),FailureKind.EMBEDDING),
    ("chroma",RetrievalError("x"),FailureKind.VECTOR_INDEX),("keyword",RuntimeError("x"),FailureKind.KEYWORD_INDEX),
    ("reranker",RuntimeError("x"),FailureKind.RERANKER),("glm",ModelInvocationError("x"),FailureKind.GLM),
    ("database",sqlite3.OperationalError("x"),FailureKind.DATABASE),("filesystem",OSError("x"),FailureKind.FILESYSTEM)])
def test_injected_failures_are_classified(stage,error,kind):
    result=FailureRecovery(1).execute(lambda:(_ for _ in ()).throw(error),stage=stage)
    assert result.failure.kind is kind

def test_retry_is_bounded_and_can_recover():
    calls=[]
    def operation():
        calls.append(1)
        if len(calls)<3:raise ModelInvocationError("temporary")
        return "ok"
    result=FailureRecovery(3).execute(operation,stage="glm")
    assert result.value=="ok" and len(calls)==3

def test_non_retryable_parser_failure_stops_immediately():
    calls=[]
    result=FailureRecovery(3).execute(lambda:(calls.append(1),(_ for _ in ()).throw(DocumentProcessingError("bad")))[1],stage="docling")
    assert len(calls)==1 and not result.failure.retryable

def test_database_retry_exhaustion_is_observable():
    result=FailureRecovery(2).execute(lambda:(_ for _ in ()).throw(sqlite3.OperationalError("locked")),stage="database")
    assert result.failure.attempts==2 and "locked" in result.failure.message
