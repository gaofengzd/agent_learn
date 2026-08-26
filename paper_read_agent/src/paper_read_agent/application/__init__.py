"""Application service boundary."""

from paper_read_agent.application.paper_service import (
    PaperService,
    UploadResult,
    UploadValidationError,
)
from paper_read_agent.application.processing_tasks import (
    ProcessingOutcome,
    ProcessingTask,
    ProcessingTaskRunner,
    ProcessingTaskStore,
    TaskStatus,
)
from paper_read_agent.application.system_health import (
    ComponentHealth,
    HealthStatus,
    SystemHealthReport,
    SystemHealthService,
)
from paper_read_agent.application.sufficiency import (
    AnswerPlan, AnswerPlanItem, SufficiencyChecker, SupportLevel,
)
from paper_read_agent.application.verification import PostGenerationVerifier, VerificationResult
from paper_read_agent.application.qa_service import PreparedEvidence, QAResult, QuestionAnsweringService
from paper_read_agent.application.summarization import SummarizationService, SummaryLevel, SummaryResult
from paper_read_agent.application.method_extraction import MethodExtractionService
from paper_read_agent.application.innovation import InnovationService, InnovationResult
from paper_read_agent.application.sessions import RestoredSession, SessionService
from paper_read_agent.application.deletion import DeletionResult, PaperDeletionService
from paper_read_agent.application.reprocessing import ReprocessingService, ReprocessResult, VersionBuild
from paper_read_agent.application.failure_recovery import FailureKind, FailureRecovery, FailureRecord
from paper_read_agent.application.resource_limits import InputLimits, ResourceGuard

__all__ = [
    "PaperService",
    "ComponentHealth",
    "HealthStatus",
    "ProcessingOutcome",
    "ProcessingTask",
    "ProcessingTaskRunner",
    "ProcessingTaskStore",
    "TaskStatus",
    "SystemHealthReport",
    "SystemHealthService",
    "UploadResult",
    "UploadValidationError",
    "AnswerPlan", "AnswerPlanItem", "SufficiencyChecker", "SupportLevel",
    "PostGenerationVerifier", "VerificationResult",
    "PreparedEvidence", "QAResult", "QuestionAnsweringService",
    "SummarizationService", "SummaryLevel", "SummaryResult", "MethodExtractionService",
    "InnovationService", "InnovationResult",
    "RestoredSession", "SessionService",
    "DeletionResult", "PaperDeletionService",
    "ReprocessingService", "ReprocessResult", "VersionBuild",
    "FailureKind", "FailureRecovery", "FailureRecord",
    "InputLimits", "ResourceGuard",
]
