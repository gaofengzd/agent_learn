"""Runtime façade that wires the local paper library to the web UI."""
from __future__ import annotations

from io import BytesIO

from paper_read_agent.application.ingestion import LocalDocumentIngestionProcessor
from paper_read_agent.application.paper_service import PaperService
from paper_read_agent.application.processing_tasks import ProcessingTaskRunner,ProcessingTaskStore,TaskStatus
from paper_read_agent.application.system_health import SystemHealthService
from paper_read_agent.config import AppSettings
from paper_read_agent.domain.models import PaperStatus
from paper_read_agent.persistence.database import SQLiteDatabase
from paper_read_agent.persistence.repositories import SQLiteDomainRepository


class LocalUIFacade:
    def __init__(self,settings:AppSettings,*,processor=None)->None:
        self.settings=settings;settings.ensure_runtime_directories()
        self.database=SQLiteDatabase(settings.storage.database_path);self.database.initialize()
        self.repository=SQLiteDomainRepository(self.database)
        self.paper_service=PaperService(self.repository,settings.storage.pdf_dir)
        self.tasks=ProcessingTaskStore(self.database)
        self.runner=ProcessingTaskRunner(
            self.repository,self.tasks,processor or LocalDocumentIngestionProcessor(settings,self.repository))

    def navigation(self):
        return {"papers":[{"title":p.title} for p in self.repository.list_papers()],"conversations":[]}

    def health(self):
        report=SystemHealthService(self.settings).check()
        return [{"name":x.name,"status":x.status.value,"message":x.message} for x in report.components]

    def list_papers(self):
        labels={PaperStatus.PENDING:"等待处理",PaperStatus.PARSING:"处理中",PaperStatus.READY:"可用",
                PaperStatus.PARTIALLY_READY:"部分可用",PaperStatus.FAILED:"处理失败"}
        values=[]
        for paper in self.repository.list_papers():
            task=self.tasks.latest_for_version(paper.active_version_id) if paper.active_version_id else None
            values.append({"paper_id":paper.paper_id,"title":paper.title,
              "status_label":labels[paper.status],"version":paper.active_version_id or "—",
              "progress":task.status.value if task and task.status in {TaskStatus.QUEUED,TaskStatus.RUNNING} else "",
              "quality":paper.quality_level.value if paper.quality_level else "",
              "missing_pages":[],"error":task.error_message if task and task.status is TaskStatus.FAILED else "",
              "readable":paper.status in {PaperStatus.READY,PaperStatus.PARTIALLY_READY},
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

    def qa_view(self):return {}
    def analysis_view(self):return {"available_papers":[],"citations":[]}
