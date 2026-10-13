"""Microsoft Agent Framework OpenTelemetry span translation."""

from deepeval.integrations.strands.instrumentator import (
    StrandsInstrumentationSettings,
    StrandsSpanInterceptor,
)
from deepeval.tracing.integrations import Integration


class MicrosoftAgentFrameworkSpanInterceptor(StrandsSpanInterceptor):
    """Translate Agent Framework GenAI semantic-convention spans."""

    integration = Integration.MICROSOFT_AGENT_FRAMEWORK
    thread_id_attribute_keys = (
        "gen_ai.conversation.id",
        "session.id",
    )


class MicrosoftAgentFrameworkInstrumentationSettings(
    StrandsInstrumentationSettings
):
    """Trace-level defaults for Microsoft Agent Framework instrumentation."""
