"""Runtime façade that wires the local paper library to the web UI."""
from __future__ import annotations

from io import BytesIO

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
        self._analysis["methods"]=rows;return rows

    def analyze_innovations(self,paper_ids):
        ids=self._paper_ids(paper_ids); result=self.reading.innovations.analyze(
            self.reading.registries(ids,QueryIntent.INNOVATION),comparison_scope=ids)
        self._remember_citations(c for x in (*result.author_claims,*result.agent_hypotheses) for c in x.citations)
        convert=lambda item,status:{"status":status,"content":item.text,"evidence_ids":[c.evidence_id for c in item.citations]}
        self._analysis["author_contributions"]=[convert(x,"作者明示") for x in result.author_claims]
        self._analysis["agent_innovations"]=[convert(x,"Agent 推断") for x in result.agent_hypotheses]
        return result

    def analysis_view(self):
        return {"available_papers":self._available_papers(),"citations":list(self._citations.values()),**self._analysis}


class _ChromaDeleteAdapter:
    def __init__(self,path):self.client=chromadb.PersistentClient(path=path)
    def delete(self,*,paper_id):
        try:collection=self.client.get_collection("paper_chunks")
        except Exception:return
        collection.delete(where={"paper_id":paper_id})
