"""Uniform failure classification and bounded retry policy."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable,TypeVar
import sqlite3
from paper_read_agent.exceptions import DocumentProcessingError,ModelInvocationError,RetrievalError
T=TypeVar("T")
class FailureKind(StrEnum):
    PARSING="parsing";OCR="ocr";EMBEDDING="embedding";VECTOR_INDEX="vector_index"
    KEYWORD_INDEX="keyword_index";RERANKER="reranker";GLM="glm";DATABASE="database";FILESYSTEM="filesystem";UNKNOWN="unknown"
@dataclass(frozen=True,slots=True)
class FailureRecord:
    kind:FailureKind;message:str;retryable:bool;attempts:int
@dataclass(frozen=True,slots=True)
class RecoveryResult:
    value:object|None;failure:FailureRecord|None
class FailureRecovery:
    RETRYABLE={FailureKind.GLM,FailureKind.VECTOR_INDEX,FailureKind.KEYWORD_INDEX,
               FailureKind.DATABASE,FailureKind.FILESYSTEM}
    def __init__(self,max_attempts:int=3):
        if max_attempts<1:raise ValueError("max_attempts must be positive")
        self.max_attempts=max_attempts
    def execute(self,operation:Callable[[],T],*,stage:str)->RecoveryResult:
        for attempt in range(1,self.max_attempts+1):
            try:return RecoveryResult(operation(),None)
            except Exception as exc:
                kind=self.classify(exc,stage);retryable=kind in self.RETRYABLE
                record=FailureRecord(kind,f"{type(exc).__name__}: {exc}",retryable,attempt)
                if not retryable or attempt==self.max_attempts:return RecoveryResult(None,record)
        raise AssertionError("unreachable")
    @staticmethod
    def classify(exc:Exception,stage:str)->FailureKind:
        stage=stage.casefold()
        for token,kind in (("ocr",FailureKind.OCR),("docling",FailureKind.PARSING),("pars",FailureKind.PARSING),
            ("embed",FailureKind.EMBEDDING),("chroma",FailureKind.VECTOR_INDEX),("vector",FailureKind.VECTOR_INDEX),
            ("keyword",FailureKind.KEYWORD_INDEX),("rerank",FailureKind.RERANKER),("glm",FailureKind.GLM),
            ("database",FailureKind.DATABASE),("file",FailureKind.FILESYSTEM)):
            if token in stage:return kind
        if isinstance(exc,sqlite3.Error):return FailureKind.DATABASE
        if isinstance(exc,OSError):return FailureKind.FILESYSTEM
        if isinstance(exc,ModelInvocationError):return FailureKind.GLM
        if isinstance(exc,RetrievalError):return FailureKind.VECTOR_INDEX
        if isinstance(exc,DocumentProcessingError):return FailureKind.PARSING
        return FailureKind.UNKNOWN
