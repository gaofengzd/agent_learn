"""Language-model integration boundary."""
from paper_read_agent.llm.glm_client import (
    Claim, GLMClient, GLMHTTPTransport, PromptRegistry, StructuredAnswer,
)

__all__ = ["Claim", "GLMClient", "GLMHTTPTransport", "PromptRegistry", "StructuredAnswer"]
