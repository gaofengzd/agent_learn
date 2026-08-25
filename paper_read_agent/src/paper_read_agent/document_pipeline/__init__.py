"""Document ingestion and normalization boundary."""

from paper_read_agent.document_pipeline.preflight import (
    EncryptedPDFError,
    InvalidPDFError,
    PagePreflight,
    PDFLimitError,
    PDFPreflight,
    PDFPreflightReport,
)
from paper_read_agent.document_pipeline.docling_parser import (
    DoclingConversionError,
    DoclingParseResult,
    DoclingParser,
)
from paper_read_agent.document_pipeline.ocr import (
    OCRDocumentResult,
    OCRLine,
    OCRPageResult,
    OCRProcessingError,
    RapidOCRProcessor,
)
from paper_read_agent.document_pipeline.normalizer import DocumentNormalizer, NormalizedDocument
from paper_read_agent.document_pipeline.quality import (
    ParseQualityEvaluator,
    QualityAssessment,
    QualityThresholds,
)
from paper_read_agent.document_pipeline.chunking import ChunkBuildResult, ParentChildChunker

__all__ = [
    "EncryptedPDFError",
    "DoclingConversionError",
    "DoclingParseResult",
    "DoclingParser",
    "OCRDocumentResult",
    "OCRLine",
    "OCRPageResult",
    "OCRProcessingError",
    "RapidOCRProcessor",
    "DocumentNormalizer",
    "NormalizedDocument",
    "ParseQualityEvaluator",
    "QualityAssessment",
    "QualityThresholds",
    "ChunkBuildResult",
    "ParentChildChunker",
    "InvalidPDFError",
    "PagePreflight",
    "PDFLimitError",
    "PDFPreflight",
    "PDFPreflightReport",
]
