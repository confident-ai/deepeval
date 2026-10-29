"""deepeval's LlamaIndex integration, built on confident-trace."""

from typing import Any, Optional

from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration
from deepeval.tracing.otel.setup import ensure_confident_trace

# confident-trace takes LlamaIndex's model calls from the provider SDKs its
# LLM classes use, so those are traced alongside it.
INSTRUMENTATIONS = (
    "llamaindex",
    "openai",
    "anthropic",
    "google_genai",
    "bedrock",
    "litellm",
)


def instrument_llama_index(dispatcher: Optional[Any] = None) -> None:
    """Trace LlamaIndex with confident-trace.

    ``dispatcher`` is accepted for compatibility: confident-trace attaches to
    LlamaIndex's root dispatcher, which every other dispatcher reports to.
    """
    with capture_tracing_integration(Integration.LLAMA_INDEX):
        ensure_confident_trace(INSTRUMENTATIONS)
