"""deepeval's Pydantic AI integration, built on confident-trace.

confident-trace reads the telemetry Pydantic AI emits once deepeval starts it
with the ``pydantic_ai`` instrumentation. ``instrument()`` shares the global
provider, which turns on every agent's telemetry when deepeval's provider is
the global one. ``DeepEvalInstrumentationSettings`` instruments one agent on
deepeval's provider, leaving the global provider to the application, and
stamps the trace settings it's given on that agent's spans.
"""

import warnings
from typing import Any, Dict, List, Optional

from opentelemetry.trace import SpanKind, Tracer

from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration
from deepeval.tracing.otel.bridge import trace_field_attributes
from deepeval.tracing.otel.setup import ensure_confident_trace, tracer_provider

try:
    from pydantic_ai.models.instrumented import InstrumentationSettings

    dependency_installed = True
except ImportError:
    dependency_installed = False
    InstrumentationSettings = object

INSTRUMENTATIONS = ("pydantic_ai",)


def is_dependency_installed() -> bool:
    if not dependency_installed:
        raise ImportError(
            "Pydantic AI is not installed. Please install it with "
            "`pip install pydantic-ai`."
        )
    return True


def instrument(every_agent: bool = True) -> None:
    is_dependency_installed()
    with capture_tracing_integration(Integration.PYDANTIC_AI):
        ensure_confident_trace(INSTRUMENTATIONS, global_provider=every_agent)


class DeepEvalInstrumentationSettings(InstrumentationSettings):
    """Pydantic AI ``InstrumentationSettings`` that trace an agent for deepeval.

    Pass them to ``Instrumentation(settings=...)`` in an agent's
    ``capabilities`` (or to ``Agent(instrument=...)`` on older Pydantic AI).

    The trace kwargs are defaults for every trace the agent produces;
    ``update_current_trace(...)`` overrides them at runtime. Span fields are
    set at the call site with ``update_current_span(...)`` or
    ``next_*_span(...)``.
    """

    def __init__(
        self,
        # Deprecated, ignored: a Confident AI platform setting. Don't build on it.
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        # Deprecated, ignored: a Confident AI platform setting. Don't build on it.
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        # Deprecated, ignored: Confident AI platform settings. Don't build on them.
        metric_collection: Optional[str] = None,
        test_case_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ):
        instrument(every_agent=False)
        super().__init__(tracer_provider=tracer_provider())
        self.tracer = _TraceDefaultsTracer(
            self.tracer,
            trace_field_attributes(
                {
                    "name": name,
                    "thread_id": thread_id,
                    "metadata": metadata,
                    "tags": tags,
                }
            ),
        )


class ConfidentInstrumentationSettings(DeepEvalInstrumentationSettings):
    """Deprecated alias for :class:`DeepEvalInstrumentationSettings`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "ConfidentInstrumentationSettings is deprecated and will be "
            "removed in a future version. Use "
            "DeepEvalInstrumentationSettings instead — same constructor.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class _TraceDefaultsTracer(Tracer):
    """Starts every span with the trace defaults; its own attributes win."""

    def __init__(self, tracer: Tracer, attributes: Dict[str, Any]):
        self._tracer = tracer
        self._attributes = attributes

    def start_span(
        self,
        name: str,
        context: Any = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self._tracer.start_span(
            name,
            context,
            kind,
            self._with_defaults(attributes),
            *args,
            **kwargs,
        )

    def start_as_current_span(
        self,
        name: str,
        context: Any = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self._tracer.start_as_current_span(
            name,
            context,
            kind,
            self._with_defaults(attributes),
            *args,
            **kwargs,
        )

    def _with_defaults(self, attributes: Any) -> Dict[str, Any]:
        return {**self._attributes, **(attributes or {})}
