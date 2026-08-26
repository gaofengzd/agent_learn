"""Safe paper deletion across indexes, files, database, and history."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable,Protocol
import json
from paper_read_agent.application.processing_tasks import ProcessingTaskStore,TaskStatus
from paper_read_agent.persistence.repositories import SQLiteDomainRepository

class DeletableIndex(Protocol):
    def delete(self,*,paper_id:str)->None:...
@dataclass(frozen=True,slots=True)
class DeletionResult:
    paper_id:str; deleted:bool; completed_steps:tuple[str,...]; errors:tuple[str,...]

class PaperDeletionService:
    def __init__(self,repository:SQLiteDomainRepository,tasks:ProcessingTaskStore,
                 vector:DeletableIndex,keyword:DeletableIndex,pdf_dir:Path,
                 *,file_remover:Callable[[Path],None]|None=None)->None:
        self.repository,self.tasks,self.vector,self.keyword=repository,tasks,vector,keyword
        self.pdf_dir=pdf_dir.resolve();self.file_remover=file_remover or (lambda path:path.unlink())
    def delete(self,paper_id:str)->DeletionResult:
        paper=self.repository.get_paper(paper_id)
        if paper is None: raise KeyError(paper_id)
        path=Path(paper.file_path).resolve()
        if path.parent!=self.pdf_dir: raise ValueError("Paper file is outside configured PDF storage")
        with self.repository.database.connect() as c:
            active=c.execute("SELECT 1 FROM processing_tasks t JOIN paper_versions v USING(version_id) WHERE v.paper_id=? AND t.status IN ('queued','running')",(paper_id,)).fetchone()
        if active: raise RuntimeError("Cannot delete a paper while processing is active")
        completed=[];errors=[]
        for name,index in (("vector",self.vector),("keyword",self.keyword)):
            try:index.delete(paper_id=paper_id);completed.append(name)
            except Exception as exc:errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if not errors:
            try:
                if path.exists():self.file_remover(path)
                completed.append("file")
            except Exception as exc:errors.append(f"file: {type(exc).__name__}: {exc}")
        if not errors:
            self._mark_history(paper_id);self.repository.delete_paper_record(paper_id);completed.extend(("history","database"))
        return DeletionResult(paper_id,not errors,tuple(completed),tuple(errors))
    def _mark_history(self,paper_id:str)->None:
        with self.repository.database.transaction() as c:
            rows=c.execute("SELECT message_id,structured_payload_json,retrieval_scope_json FROM messages").fetchall()
            for row in rows:
                scope=json.loads(row["retrieval_scope_json"])
                if paper_id not in scope.get("paper_ids",[]):continue
                payload=json.loads(row["structured_payload_json"]);deleted=payload.setdefault("deleted_paper_ids",[])
                if paper_id not in deleted:deleted.append(paper_id)
                c.execute("UPDATE messages SET structured_payload_json=? WHERE message_id=?",(json.dumps(payload),row["message_id"]))
