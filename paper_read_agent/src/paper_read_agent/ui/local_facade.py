"""Runtime façade that wires the local paper library to the web UI."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from io import BytesIO
import json
from threading import Lock
from uuid import uuid4

import chromadb

from paper_read_agent.application.ingestion import LocalDocumentIngestionProcessor
from paper_read_agent.application.deletion import PaperDeletionService
from paper_read_agent.application.paper_service import PaperService
from paper_read_agent.application.processing_tasks import ProcessingTaskRunner,ProcessingTaskStore,TaskStatus
from paper_read_agent.application.system_health import SystemHealthService
from paper_read_agent.application.local_reading import LocalReadingServices
from paper_read_agent.application.sessions import SessionService
from paper_read_agent.application.summarization import SummaryLevel
from paper_read_agent.retrieval.query_planner import QueryIntent
from paper_read_agent.retrieval.keyword_index import SQLiteKeywordIndex
from paper_read_agent.config import AppSettings
from paper_read_agent.domain.models import PaperStatus
from paper_read_agent.persistence.database import SQLiteDatabase
from paper_read_agent.persistence.repositories import SQLiteDomainRepository


class _AnalysisTaskQueue:
    """SQLite-backed queue that keeps slow analysis work outside HTTP requests."""

    def __init__(self, execute, database, *, max_workers=1):
        self.execute = execute
        self.database = database
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="paper-analysis"
        )
        self._lock = Lock()
        self._futures: dict[str, Future] = {}
        now = datetime.now(UTC).isoformat()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE analysis_tasks SET status='failed',"
                "message='服务重启，未完成任务已中断',finished_at=?,updated_at=? "
                "WHERE status IN ('queued','running')", (now, now)
            )

    def submit(self, operation, paper_ids, version_ids, level="standard"):
        now = datetime.now(UTC).isoformat()
        paper_json=json.dumps(list(paper_ids))
        version_json=json.dumps(list(version_ids))
        with self.database.connect() as connection:
            existing=connection.execute(
                "SELECT * FROM analysis_tasks WHERE operation=? AND paper_ids_json=? "
                "AND version_ids_json=? AND level=? AND status IN ('queued','running') "
                "ORDER BY created_at DESC LIMIT 1",
                (operation,paper_json,version_json,level)
            ).fetchone()
        if existing is not None:
            return {"task_id":existing["task_id"],"operation":operation,
                    "status":existing["status"],"paper_ids":list(paper_ids),
                    "level":level,"message":existing["message"],"duplicate":True,
                    "created_at":existing["created_at"],"updated_at":existing["updated_at"]}
        task_id = str(uuid4())
        task = {"task_id": task_id, "operation": operation, "status": "queued",
                "paper_ids": list(paper_ids), "level": level, "message": "等待执行",
                "created_at": now, "updated_at": now}
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO analysis_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id,operation,paper_json,version_json,level,"queued","等待执行",
                 0,None,now,None,None,now)
            )
        with self._lock:
            self._futures[task_id] = self._executor.submit(
                self._run, task_id, operation, tuple(paper_ids), level
            )
        return dict(task)

    def cancel(self,task_id):
        now=datetime.now(UTC).isoformat()
        with self.database.transaction() as connection:
            row=connection.execute(
                "SELECT status FROM analysis_tasks WHERE task_id=?",(task_id,)
            ).fetchone()
            if row is None:raise KeyError(task_id)
            if row["status"] not in {"queued","running"}:
                raise RuntimeError("Only an active analysis task can be cancelled")
            connection.execute(
                "UPDATE analysis_tasks SET cancel_requested=1,"
                "message='正在取消',updated_at=? WHERE task_id=?",(now,task_id)
            )
        with self._lock:
            future=self._futures.get(task_id)
            cancelled=bool(future and future.cancel())
        if cancelled:
            self._set(task_id,status="cancelled",message="任务已取消",
                      finished_at=datetime.now(UTC).isoformat())
        return self.snapshot_task(task_id)

    def snapshot_task(self,task_id):
        return next((task for task in self.snapshot() if task["task_id"]==task_id),None)

    def is_cancel_requested(self,task_id):
        with self.database.connect() as connection:
            row=connection.execute(
                "SELECT cancel_requested FROM analysis_tasks WHERE task_id=?",(task_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def snapshot(self):
        with self.database.connect() as connection:
            rows=connection.execute(
                "SELECT * FROM analysis_tasks ORDER BY created_at DESC"
            ).fetchall()
        return tuple({
            "task_id":row["task_id"],"operation":row["operation"],
            "paper_ids":json.loads(row["paper_ids_json"]),"level":row["level"],
            "status":row["status"],"message":row["message"],
            "cancel_requested":bool(row["cancel_requested"]),
            "created_at":row["created_at"],"updated_at":row["updated_at"],
        } for row in rows)

    def successful_payloads(self):
        with self.database.connect() as connection:
            rows=connection.execute(
                "SELECT result_json,paper_ids_json,version_ids_json FROM analysis_tasks "
                "WHERE status='succeeded' AND result_json IS NOT NULL "
                "ORDER BY created_at"
            ).fetchall()
        values=[]
        for row in rows:
            payload=json.loads(row["result_json"])
            payload["_paper_ids"]=json.loads(row["paper_ids_json"])
            payload["_version_ids"]=json.loads(row["version_ids_json"])
            values.append(payload)
        return tuple(values)

    def future(self, task_id):
        with self._lock:
            return self._futures.get(task_id)

    def shutdown(self):
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _set(self, task_id, **changes):
        changes["updated_at"] = datetime.now(UTC).isoformat()
        allowed={"status","message","result_json","started_at","finished_at",
                 "updated_at","cancel_requested"}
        if any(name not in allowed for name in changes):raise ValueError("Invalid task update")
        assignments=",".join(f"{name}=?" for name in changes)
        with self.database.transaction() as connection:
            connection.execute(
                f"UPDATE analysis_tasks SET {assignments} WHERE task_id=?",
                (*changes.values(),task_id)
            )

    def _run(self, task_id, operation, paper_ids, level):
        if self.is_cancel_requested(task_id):
            self._set(task_id,status="cancelled",message="任务已取消",
                      finished_at=datetime.now(UTC).isoformat())
            return
        started=datetime.now(UTC).isoformat()
        self._set(task_id,status="running",message="正在检索、重排并生成",
                  started_at=started)
        try:
            payload=self.execute(
                operation,paper_ids,level,lambda:self.is_cancel_requested(task_id)
            )
        except Exception as exc:
            finished=datetime.now(UTC).isoformat()
            if self.is_cancel_requested(task_id):
                self._set(task_id,status="cancelled",message="任务已取消",
                          finished_at=finished)
            else:
                self._set(task_id,status="failed",message=f"{type(exc).__name__}: {exc}",
                          finished_at=finished)
        else:
            finished=datetime.now(UTC).isoformat()
            if self.is_cancel_requested(task_id):
                self._set(task_id,status="cancelled",message="任务已取消",
                          finished_at=finished)
            else:
                self._set(task_id,status="succeeded",message="分析已完成",
                          result_json=json.dumps(payload,ensure_ascii=False),
                          finished_at=finished)


class LocalUIFacade:
    def __init__(self,settings:AppSettings,*,processor=None,reading_services=None)->None:
        self.settings=settings;settings.ensure_runtime_directories()
        self.database=SQLiteDatabase(settings.storage.database_path);self.database.initialize()
        self.repository=SQLiteDomainRepository(self.database)
        self.paper_service=PaperService(self.repository,settings.storage.pdf_dir)
        self.tasks=ProcessingTaskStore(self.database)
        self.sessions=SessionService(self.repository)
        self._reading_services=reading_services
        self._analysis={}
        self._citations={}
        self._result_lock=Lock()
        self._analysis_queue=_AnalysisTaskQueue(self._execute_analysis,self.database)
        self._restore_analysis()
        self.runner=ProcessingTaskRunner(
            self.repository,self.tasks,processor or LocalDocumentIngestionProcessor(settings,self.repository))

    @property
    def reading(self):
        if self._reading_services is None:
            self._reading_services=LocalReadingServices(self.settings,self.repository)
        return self._reading_services

    def navigation(self):
        return {"papers":[{"paper_id":p.paper_id,"title":p.title} for p in self.repository.list_papers()],
                "conversations":[{"conversation_id":c.conversation_id,"title":c.title}
                                 for c in self.repository.list_conversations()]}

    def health(self):
        report=SystemHealthService(self.settings).check()
        return [{"name":x.name,"status":x.status.value,"message":x.message} for x in report.components]

    def list_papers(self):
        labels={PaperStatus.PENDING:"等待处理",PaperStatus.PARSING:"处理中",PaperStatus.READY:"可用",
                PaperStatus.PARTIALLY_READY:"部分可用",PaperStatus.FAILED:"处理失败"}
        values=[]
        for paper in self.repository.list_papers():
            task=self.tasks.latest_for_version(paper.active_version_id) if paper.active_version_id else None
            failed=paper.status is PaperStatus.FAILED
            values.append({"paper_id":paper.paper_id,"title":paper.title,
              "status_label":labels[paper.status],"version":paper.active_version_id or "—",
              "progress":task.status.value if task and task.status in {TaskStatus.QUEUED,TaskStatus.RUNNING} else "",
              "quality":"failed" if failed else (paper.quality_level.value if paper.quality_level else ""),
              "missing_pages":[],"error":task.error_message if task and task.status is TaskStatus.FAILED else "",
              "readable":paper.status in {PaperStatus.READY,PaperStatus.PARTIALLY_READY},
              "unreadable_label":"处理失败，暂不可阅读" if failed else "处理中，暂不可阅读",
              "can_retry":bool(task and task.status is TaskStatus.FAILED)})
        return values

    def upload_papers(self,files):
        results=[]
        for filename,content_type,payload in files:
            result=self.paper_service.upload_pdf(
                original_filename=filename,content_type=content_type,stream=BytesIO(payload))
            if result.version is not None:self.runner.enqueue(result.version.version_id)
            results.append({"paper_id":result.paper.paper_id,"duplicate":result.duplicate})
        return results

    def retry_paper(self,paper_id):
        paper=self.repository.get_paper(paper_id)
        if paper is None or not paper.active_version_id:raise KeyError(paper_id)
        self.runner.retry(paper.active_version_id)

    def reprocess_paper(self,paper_id):
        paper=self.repository.get_paper(paper_id)
        if paper is None or not paper.active_version_id:raise KeyError(paper_id)
        latest=self.tasks.latest_for_version(paper.active_version_id)
        if latest and latest.status in {TaskStatus.QUEUED,TaskStatus.RUNNING}:raise RuntimeError("论文正在处理中")
        self.runner.enqueue(paper.active_version_id)

    def delete_paper(self,paper_id):
        vector=_ChromaDeleteAdapter(self.settings.storage.chroma_dir)
        keyword=SQLiteKeywordIndex(self.settings.storage.database_path)
        result=PaperDeletionService(self.repository,self.tasks,vector,keyword,
                                    self.settings.storage.pdf_dir).delete(paper_id)
        if not result.deleted:raise RuntimeError("；".join(result.errors))
        return result

    def _available_papers(self):
        return [{"paper_id":p.paper_id,"title":p.title} for p in self.repository.list_papers()
                if p.status in {PaperStatus.READY,PaperStatus.PARTIALLY_READY}]

    def create_session(self,title):return self.sessions.create(title,"library")
    def delete_session(self,conversation_id):return self.sessions.delete(conversation_id)
    def change_scope(self,conversation_id,scope_mode,paper_ids):
        return self.sessions.change_scope(conversation_id,scope_mode,paper_ids)
    def ask(self,conversation_id,question):
        restored=self.sessions.restore(conversation_id); available=[p["paper_id"] for p in self._available_papers()]
        questions=[m.content for m in restored.messages if m.role.value=="user"]
        result=self.reading.qa.answer(question,conversation_id=conversation_id,
            scope_mode=restored.conversation.scope_mode,
            paper_ids=restored.conversation.selected_paper_ids if restored.conversation.scope_mode=="selected" else available,
            conversation_questions=questions)
        self._remember_citations(result.citations);return result

    def _remember_citations(self,citations):
        with self._result_lock:
            for item in citations:self._citations[item.evidence_id]={"evidence_id":item.evidence_id,
                "label":item.label,"excerpt":item.excerpt,"source_type":item.source_type}

    def qa_view(self):
        conversations=self.repository.list_conversations()
        if not conversations:return {}
        restored=self.sessions.restore(conversations[0].conversation_id); conv=restored.conversation
        papers={p["paper_id"]:p["title"] for p in self._available_papers()}; citations=[]; messages=[]
        status_labels={"answered":"充分","partially_answered":"部分回答","conflicted":"存在冲突",
                       "insufficient_evidence":"证据不足","document_quality_failure":"文档质量不足","out_of_scope":"范围不明确"}
        for message in restored.messages:
            payload=message.structured_payload or {}
            messages.append({"role_label":"用户" if message.role.value=="user" else "助手","content":message.content,
                "answer_status":status_labels.get(message.answer_status.value,"") if message.answer_status else "",
                "unanswered_items":payload.get("unanswered_items",[]),"evidence_ids":list(message.evidence_ids)})
        return {"conversation_id":conv.conversation_id,"title":conv.title,"paper_ids":list(conv.selected_paper_ids),
            "scope_label":"全库" if conv.scope_mode=="library" else "、".join(papers.get(x,x) for x in conv.selected_paper_ids),
            "available_papers":self._available_papers(),"messages":messages,"citations":list(self._citations.values())}

    def _paper_ids(self,paper_ids):
        available={p["paper_id"] for p in self._available_papers()}; values=tuple(dict.fromkeys(paper_ids))
        if not values:raise ValueError("请至少选择一篇可用论文")
        if any(x not in available for x in values):raise ValueError("论文不在可用范围内")
        return values

    def summarize(self,paper_ids,level):
        ids=self._paper_ids(paper_ids); mapped={"brief":SummaryLevel.QUICK,"quick":SummaryLevel.QUICK,
            "standard":SummaryLevel.STANDARD,"detailed":SummaryLevel.DETAILED}[level]
        result=self.reading.summary.summarize(mapped,self.reading.registries(ids,QueryIntent.SUMMARY))
        self._remember_citations(c for x in result.sections for c in x.citations)
        with self._result_lock:
            self._analysis["summary"]={"level_label":{"quick":"简要","standard":"标准","detailed":"详细"}[mapped.value],
                "scope_label":"、".join(ids),"content":"\n".join(x.text for x in result.sections),
                "evidence_ids":[c.evidence_id for x in result.sections for c in x.citations]}
        return result

    def extract_methods(self,paper_ids):
        ids=self._paper_ids(paper_ids); rows=[]
        for registry in self.reading.registries(ids,QueryIntent.METHOD).values():
            for method in self.reading.methods.extract(registry):
                for field in method.fields:
                    self._remember_citations(field.citations)
                    rows.append({"field":field.name,"role":method.role.value,
                        "status":field.source_status.value,"content":field.value or "未说明",
                        "evidence_ids":[c.evidence_id for c in field.citations]})
        with self._result_lock:self._analysis["methods"]=rows
        return rows

    def analyze_innovations(self,paper_ids):
        ids=self._paper_ids(paper_ids); result=self.reading.innovations.analyze(
            self.reading.registries(ids,QueryIntent.INNOVATION),comparison_scope=ids)
        self._remember_citations(c for x in (*result.author_claims,*result.agent_hypotheses) for c in x.citations)
        convert=lambda item,status:{"status":status,"content":item.text,"evidence_ids":[c.evidence_id for c in item.citations]}
        with self._result_lock:
            self._analysis["author_contributions"]=[convert(x,"作者明示") for x in result.author_claims]
            self._analysis["agent_innovations"]=[convert(x,"Agent 推断") for x in result.agent_hypotheses]
        return result

    def submit_analysis(self,operation,paper_ids,level="standard"):
        ids=self._paper_ids(paper_ids)
        if operation not in {"summary","methods","innovations"}:raise ValueError("未知分析类型")
        if operation=="summary" and level not in {"brief","quick","standard","detailed"}:
            raise ValueError("未知总结层级")
        versions=tuple(self.repository.get_paper(pid).active_version_id for pid in ids)
        return self._analysis_queue.submit(operation,ids,versions,level)

    def cancel_analysis(self,task_id):
        return self._analysis_queue.cancel(task_id)

    def _execute_analysis(self,operation,paper_ids,level,cancellation_check):
        self.reading.set_analysis_cancellation_check(cancellation_check)
        try:
            if operation=="summary":
                self.summarize(paper_ids,level); keys=("summary",)
            elif operation=="methods":
                self.extract_methods(paper_ids); keys=("methods",)
            elif operation=="innovations":
                self.analyze_innovations(paper_ids)
                keys=("author_contributions","agent_innovations")
            else:raise ValueError("未知分析类型")
            with self._result_lock:
                return {"analysis":{key:self._analysis.get(key) for key in keys},
                        "citations":list(self._citations.values())}
        finally:
            self.reading.set_analysis_cancellation_check(lambda:False)

    def _restore_analysis(self):
        for payload in self._analysis_queue.successful_payloads():
            paper_ids=payload.pop("_paper_ids",())
            version_ids=payload.pop("_version_ids",())
            current=[self.repository.get_paper(pid) for pid in paper_ids]
            if len(current)!=len(version_ids) or any(
                paper is None or paper.active_version_id!=version_id
                for paper,version_id in zip(current,version_ids,strict=True)
            ):
                continue
            self._analysis.update(payload.get("analysis",{}))
            for item in payload.get("citations",()):
                self._citations[item["evidence_id"]]=item

    def analysis_view(self):
        tasks=self._analysis_queue.snapshot()
        with self._result_lock:
            citations=list(self._citations.values())
            analysis=dict(self._analysis)
        return {"available_papers":self._available_papers(),
                "citations":citations,"tasks":tasks,
                "analysis_active":any(x["status"] in {"queued","running"} for x in tasks),
                **analysis}


class _ChromaDeleteAdapter:
    def __init__(self,path):self.client=chromadb.PersistentClient(path=path)
    def delete(self,*,paper_id):
        try:collection=self.client.get_collection("paper_chunks")
        except Exception:return
        collection.delete(where={"paper_id":paper_id})
