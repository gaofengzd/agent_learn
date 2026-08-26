"""Complete local PDF-to-index ingestion orchestration."""
from __future__ import annotations

from dataclasses import replace

from paper_read_agent.application.processing_tasks import ProcessingOutcome
from paper_read_agent.config import AppSettings
from paper_read_agent.document_pipeline.chunking import ParentChildChunker
from paper_read_agent.document_pipeline.docling_parser import DoclingParser
from paper_read_agent.document_pipeline.normalizer import DocumentNormalizer
from paper_read_agent.document_pipeline.ocr import RapidOCRProcessor
from paper_read_agent.document_pipeline.preflight import PDFPreflight
from paper_read_agent.document_pipeline.quality import ParseQualityEvaluator
from paper_read_agent.domain.models import PaperStatus, QualityLevel
from paper_read_agent.persistence.repositories import SQLiteDomainRepository
from paper_read_agent.retrieval.keyword_index import SQLiteKeywordIndex
from paper_read_agent.retrieval.vector_index import ChromaVectorIndex, LocalBGEEmbedder, VectorRecord


class LocalDocumentIngestionProcessor:
    """Lazily load heavy local models and persist one paper version end to end."""

    def __init__(self, settings: AppSettings, repository: SQLiteDomainRepository) -> None:
        self.settings = settings
        self.repository = repository
        self._components: tuple | None = None

    def __call__(self, version_id: str) -> ProcessingOutcome:
        version = self.repository.get_version(version_id)
        if version is None:raise KeyError(version_id)
        paper = self.repository.get_paper(version.paper_id)
        if paper is None:raise KeyError(version.paper_id)
        preflight,parser,ocr,chunker,vector,keyword=self._load_components()
        report=preflight.inspect(paper.file_path)
        parsed=parser.parse(paper.file_path,version_id=version_id)
        ocr_result=ocr.process(paper.file_path,report,version_id=version_id,render_page=preflight.render_page)
        document=DocumentNormalizer().normalize(
            version_id=version_id,docling_document=parsed.document,
            preflight=report,ocr=ocr_result)
        assessment=ParseQualityEvaluator().evaluate(document,report,ocr_result)
        ParseQualityEvaluator().persist(self.repository,assessment)
        # Neighbor links may point forward, so insert all rows before restoring
        # their self-referential foreign keys.
        for block in document.blocks:
            self.repository.create_block(replace(block,previous_block_id=None,next_block_id=None))
        for block in document.blocks:
            if block.previous_block_id or block.next_block_id:self.repository.update_block(block)
        built=chunker.build(document)
        for chunk in built.chunks:self.repository.create_chunk(chunk)
        records=[VectorRecord(chunk,paper.paper_id) for chunk in built.chunks]
        keyword.upsert(records);vector.upsert(records)
        current_version=self.repository.get_version(version_id)
        current_paper=self.repository.get_paper(paper.paper_id)
        assert current_version is not None and current_paper is not None
        self.repository.update_version(replace(
            current_version,parser_name=parsed.parser_name,parser_version=parsed.parser_version,
            ocr_name=ocr_result.engine_name,ocr_version=ocr_result.engine_version,
            embedding_model_id=vector.embedder.model_id))
        self.repository.update_paper(replace(
            current_paper,page_count=report.page_count,active_version_id=version_id))
        status={QualityLevel.READY:PaperStatus.READY,
                QualityLevel.PARTIALLY_READY:PaperStatus.PARTIALLY_READY,
                QualityLevel.FAILED:PaperStatus.FAILED}[assessment.report.final_level]
        if status is PaperStatus.FAILED:
            raise RuntimeError("Document parsing quality is below the usable threshold")
        return ProcessingOutcome(status,assessment.report.final_level)

    def _load_components(self) -> tuple:
        if self._components is None:
            settings=self.settings
            preflight=PDFPreflight()
            parser=DoclingParser(settings.storage.parsed_dir)
            ocr=RapidOCRProcessor(settings.storage.parsed_dir)
            chunker=ParentChildChunker.from_local_model(
                settings.chunks,settings.models.embedding_model_path)
            embedder=LocalBGEEmbedder(settings.models.embedding_model_path)
            vector=ChromaVectorIndex(settings.storage.chroma_dir,embedder)
            keyword=SQLiteKeywordIndex(settings.storage.database_path)
            self._components=(preflight,parser,ocr,chunker,vector,keyword)
        return self._components
