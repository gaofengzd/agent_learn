"""Isolated reprocessing and atomic active-version switching."""
from __future__ import annotations
from dataclasses import dataclass,replace
from hashlib import sha256
from threading import Lock
from typing import Protocol
from uuid import uuid4
from paper_read_agent.domain.models import PaperStatus,PaperVersion
from paper_read_agent.persistence.repositories import SQLiteDomainRepository

@dataclass(frozen=True,slots=True)
class VersionBuild:
    parser_name:str;parser_version:str;ocr_name:str|None;ocr_version:str|None
    embedding_model_id:str;chunk_version:str
class VersionProcessor(Protocol):
    def process(self,version_id:str)->VersionBuild:...
@dataclass(frozen=True,slots=True)
class ReprocessResult:
    version_id:str;activated:bool;error:str|None=None

class ReprocessingService:
    def __init__(self,repository:SQLiteDomainRepository,processor:VersionProcessor,
                 *,expected_embedding_model_id:str,expected_chunk_version:str)->None:
        self.repository,self.processor=repository,processor
        self.expected_embedding_model_id,self.expected_chunk_version=expected_embedding_model_id,expected_chunk_version
        self._locks:dict[str,Lock]={}
    def reprocess(self,paper_id:str)->ReprocessResult:
        paper=self.repository.get_paper(paper_id)
        if paper is None:raise KeyError(paper_id)
        lock=self._locks.setdefault(paper_id,Lock())
        if not lock.acquire(blocking=False):raise RuntimeError("Reprocessing is already active for this paper")
        version_id=str(uuid4());fingerprint=sha256(f"{paper.content_hash}:{version_id}".encode()).hexdigest()
        version=self.repository.create_version(PaperVersion(version_id,paper_id,fingerprint,parse_status=PaperStatus.PARSING))
        try:
            build=self.processor.process(version_id)
            if build.embedding_model_id!=self.expected_embedding_model_id:raise ValueError("Embedding model version mismatch")
            if build.chunk_version!=self.expected_chunk_version:raise ValueError("Chunk version mismatch")
            with self.repository.database.transaction() as c:
                c.execute("UPDATE paper_versions SET parser_name=?,parser_version=?,ocr_name=?,ocr_version=?,embedding_model_id=?,parse_status='ready',updated_at=CURRENT_TIMESTAMP WHERE version_id=?",
                          (build.parser_name,build.parser_version,build.ocr_name,build.ocr_version,build.embedding_model_id,version_id))
                c.execute("UPDATE papers SET active_version_id=?,status='ready',quality_level='ready',updated_at=CURRENT_TIMESTAMP WHERE paper_id=?",
                          (version_id,paper_id))
            return ReprocessResult(version_id,True)
        except Exception as exc:
            current=self.repository.get_version(version_id)
            if current:self.repository.update_version(replace(current,parse_status=PaperStatus.FAILED))
            return ReprocessResult(version_id,False,f"{type(exc).__name__}: {exc}")
        finally:lock.release()
