"""Project-level exception hierarchy."""


class PaperAgentError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(PaperAgentError):
    """Raised when application configuration is missing or invalid."""


class ExternalDependencyError(PaperAgentError):
    """Raised when a configured external dependency cannot be used."""


class DocumentProcessingError(PaperAgentError):
    """Raised when a document-processing stage fails."""


class RetrievalError(PaperAgentError):
    """Raised when a retrieval stage fails."""


class ModelInvocationError(PaperAgentError):
    """Raised when a language-model request fails."""
