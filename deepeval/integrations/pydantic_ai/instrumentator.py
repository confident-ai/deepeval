from __future__ import annotations

import contextvars
import logging
import warnings
from time import perf_counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.integrations.otel_instrumentation.utils import (
    bridge_otel_root_to_deepeval_parent,
    finalize_span_placeholder,
    pop_implicit_trace_context,
    push_implicit_trace_context,
    serialize_trace_context_to_otel_attrs,
)
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.context import (
    apply_pending_to_span,
    current_span_context,
    pop_pending_for,
)
from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.context_aware_processor import (
    ContextAwareSpanProcessor,
)
from deepeval.tracing.otel.utils import (
    set_span_attribute_post_end,
    to_hex_string,
)
from deepeval.tracing.perf_epoch_bridge import init_clock_bridge
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    SpanType,
    Trace,
    TraceSpanStatus,
)
from deepeval.tracing.integrations import Integration
from deepeval.tracing.utils import (
    infer_provider_from_model,
    normalize_span_provider_for_platform,
)

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    # Optional dependencies
    from opentelemetry.sdk.trace import (
        ReadableSpan as _ReadableSpan,
        SpanProcessor as _SpanProcessor,
        TracerProvider,
    )
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.trace import set_tracer_provider
    from pydantic_ai.models.instrumented import (
        InstrumentationSettings as _BaseInstrumentationSettings,
    )

    dependency_installed = True
except ImportError as e:
    dependency_installed = False

    # Preserve previous behavior: only log when verbose mode is enabled.
    if settings.DEEPEVAL_VERBOSE_MODE:
        if isinstance(e, ModuleNotFoundError):
            logger.warning(
                "Optional tracing dependency not installed: %s",
                getattr(e, "name", repr(e)),
                stacklevel=2,
            )
        else:
            logger.warning(
                "Optional tracing import failed: %s",
                e,
                stacklevel=2,
            )

    # Dummy fallbacks so imports and class definitions don't crash when
    # optional deps are missing. Actual use is still guarded by
    # is_dependency_installed().
    class _BaseInstrumentationSettings:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _SpanProcessor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_start(self, span: Any, parent_context: Any) -> None:
            pass

        def on_end(self, span: Any) -> None:
            pass

    class _ReadableSpan:
        pass


def is_dependency_installed() -> bool:
    if not dependency_installed:
        raise ImportError(
            "Dependencies are not installed. Please install it with "
            "`pip install pydantic-ai opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http`."
        )
    return True


if TYPE_CHECKING:
    # For type checkers, use real types
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
    from pydantic_ai.models.instrumented import InstrumentationSettings
else:
    # At runtime we always have something to subclass / annotate with
    InstrumentationSettings = _BaseInstrumentationSettings
    SpanProcessor = _SpanProcessor
    ReadableSpan = _ReadableSpan

# Routing + OTLP endpoint live in ContextAwareSpanProcessor now.
init_clock_bridge()  # initialize clock bridge for perf_counter() to epoch_nanos conversion


class DeepEvalInstrumentationSettings(InstrumentationSettings):
    """Pydantic AI ``InstrumentationSettings`` that wires deepeval's OTel
    pipeline.

    Construction does the non-negotiable plumbing — creates a
    ``TracerProvider``, registers ``SpanInterceptor`` and
    ``ContextAwareSpanProcessor``, sets the global tracer provider, and
    forwards itself to ``Agent(instrument=...)``. The constructor is
    required for the integration to work; you cannot use the runtime
    helpers (``update_current_trace`` / ``update_current_span``) to
    bootstrap the OTel pipeline.

    Trace-level kwargs (``name``, ``thread_id``, ``user_id``,
    ``metadata``, ``tags``, ``metric_collection``, ``test_case_id``,
    ``turn_id``) are convenience defaults stamped onto every trace
    produced by this agent. They are ALWAYS overridable at runtime via
    ``update_current_trace(...)`` from anywhere in the call stack — the
    runtime call wins on any field it touches. Settings defaults exist
    purely to save boilerplate when every trace from this agent should
    carry the same value.

    Span-level configuration intentionally lives only at the call site:
    use ``update_current_span(metric_collection=..., metadata=..., ...)``
    from inside your tool / agent body. The span placeholder pushed by
    ``SpanInterceptor.on_start`` is the write target.

    A Confident AI ``api_key`` is fully optional. When omitted (and
    ``CONFIDENT_API_KEY`` isn't in the environment), the OTel pipeline
    still runs locally — spans are produced and the ``SpanInterceptor``
    still translates them into ``confident.*`` attributes — but no
    ``x-confident-api-key`` header is attached to the OTLP exporter, so
    the Confident AI backend will reject the upload. Wire a key whenever
    you actually want traces to land in Confident AI; otherwise this
    class is fine to use as a pure local OTel instrumentation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        metric_collection: Optional[str] = None,
        test_case_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ):
        is_dependency_installed()

        if trace_manager.environment is not None:
            _environment = trace_manager.environment
        elif settings.CONFIDENT_TRACE_ENVIRONMENT is not None:
            _environment = settings.CONFIDENT_TRACE_ENVIRONMENT
        else:
            _environment = "development"
        if _environment and _environment in [
            "production",
            "staging",
            "development",
            "testing",
        ]:
            self.environment = _environment

        self.name = name
        self.thread_id = thread_id
        self.user_id = user_id
        self.metadata = metadata
        self.tags = tags
        self.metric_collection = metric_collection
        self.test_case_id = test_case_id
        self.turn_id = turn_id

        # Resolve api_key from env if not supplied. May still be None —
        # we deliberately do NOT raise. The OTel pipeline is still useful
        # without a Confident AI key (local span generation, attribute
        # translation, ContextAwareSpanProcessor routing); only the
        # outbound auth header is gated on the key being present.
        if not api_key:
            api_key = get_confident_api_key()

        trace_provider = TracerProvider()

        # Per-span attribute writes (thread/user/tags/metric_collection lookups
        # against the live deepeval contexts) happen here.
        span_interceptor = SpanInterceptor(self)
        trace_provider.add_span_processor(span_interceptor)

        # Single processor handles both transports: REST (via
        # ConfidentSpanExporter -> trace_manager) when a deepeval trace
        # context is active or an evaluation is running, OTLP otherwise.
        trace_provider.add_span_processor(
            ContextAwareSpanProcessor(api_key=api_key)
        )

        try:
            set_tracer_provider(trace_provider)
        except Exception as e:
            # Handle case where provider is already set (optional warning)
            logger.warning(f"Could not set global tracer provider: {e}")

        super().__init__(tracer_provider=trace_provider)


class ConfidentInstrumentationSettings(DeepEvalInstrumentationSettings):
    """Deprecated alias for :class:`DeepEvalInstrumentationSettings`.

    The original name implied a Confident AI account was required. Now
    that the API key is fully optional, the class is named after the SDK
    that owns it (``deepeval``) rather than the cloud product it
    optionally uploads to. Use ``DeepEvalInstrumentationSettings``
    directly in new code; this alias remains for backward compatibility
    and will be removed in a future release.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "ConfidentInstrumentationSettings is deprecated and will be "
            "removed in a future version. Use "
            "DeepEvalInstrumentationSettings instead — same constructor, "
            "and a Confident AI api_key is now optional.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class SpanInterceptor(SpanProcessor):
    """Translate Pydantic AI OTel spans into deepeval ``confident.*`` attrs.

    Trace-level attrs (``confident.trace.*``) are resolved per-span as a
    union of the live ``current_trace_context`` (mutated anywhere via
    ``update_current_trace(...)``) and the ``DeepEvalInstrumentationSettings``
    trace defaults (``name``, ``thread_id``, ``user_id``, ``tags``,
    ``metadata``, ``metric_collection``, ``test_case_id``, ``turn_id``)
    — context wins on any field it touches, settings fall back.

    Span-level attrs (``confident.span.*``) are populated EXCLUSIVELY from
    a per-OTel-span ``BaseSpan`` placeholder pushed onto
    ``current_span_context`` for the span's lifetime. This is what makes
    ``update_current_span(metadata=..., name=..., input=..., output=...,
    metric_collection=..., ...)`` work from anywhere in the call stack —
    including from inside ``@agent.tool_plain`` functions — just like
    Langfuse's SDK. At ``on_end`` the placeholder's mutated fields are
    serialized back into ``confident.span.*`` OTel attributes so the
    exporter (REST or OTLP) picks them up.
    ``DeepEvalInstrumentationSettings`` carries no span-level fields by
    design — span configuration is a runtime concern.
    """

    LLM_OPERATION_NAMES = {"chat", "generate_content", "text_completion"}

    def __init__(self, settings_instance: DeepEvalInstrumentationSettings):
        self.settings = settings_instance
        # Per-OTel-span state, keyed by span_id. Two spans never share an id
        # within a process so this is safe across threads / asyncio tasks.
        self._tokens: Dict[int, contextvars.Token] = {}
        self._placeholders: Dict[int, BaseSpan] = {}
        # Per-OTel-root-span state for the implicit trace placeholder we
        # push when there's no enclosing ``@observe`` / ``with trace(...)``
        # context. Keyed by the root span's ``span_id`` so we know to clean
        # up when that exact span ends.
        self._trace_tokens: Dict[int, contextvars.Token] = {}
        self._trace_placeholders: Dict[int, Trace] = {}

    def on_start(self, span, parent_context):
        # NOTE: we deliberately do NOT mutate ``trace_ctx.uuid`` to match the
        # OTel trace_id here. Doing so would desync ``trace.uuid`` from its
        # ``trace_manager.active_traces`` dict key, causing the exporter to
        # cache-miss on lookup and spawn a phantom duplicate trace.
        # ``ConfidentSpanExporter`` re-keys incoming OTel spans to the active
        # context's real trace_uuid when a deepeval trace is in scope.

        # Trace-level + span-level user-mutable attrs (everything that
        # ``update_current_trace(...)`` / ``update_current_span(...)`` can
        # change) are written at ``on_end`` instead of here, so the OTel span
        # captures the LATEST values rather than a stale on_start snapshot.
        # See ``_serialize_trace_context_to_otel_attrs`` and
        # ``serialize_placeholder_to_otel_attrs``.

        # ----- push implicit trace context for bare agent.run callers -----
        # If the caller didn't wrap in ``@observe`` / ``with trace(...)`` and
        # this is the OTel root span, push an implicit ``Trace`` placeholder
        # onto ``current_trace_context`` so ``update_current_trace(...)``
        # from inside tools / nested helpers actually mutates something.
        # The placeholder is tagged ``_is_otel_implicit=True`` so that
        # ``ContextAwareSpanProcessor`` keeps routing to OTLP (caller didn't
        # opt into REST). Mutations are picked up automatically by the
        # existing per-span ``_serialize_trace_context_to_otel_attrs`` since
        # it reads from ``current_trace_context`` at every ``on_end``.
        push_implicit_trace_context(
            span, self._trace_tokens, self._trace_placeholders
        )

        # ----- bridge OTel root span to enclosing deepeval span -----
        # When an OTel root span starts inside a deepeval-managed span (the
        # canonical case being ``@observe(type="agent") -> agent.run(...)``),
        # OTel sees no parent and the exporter would otherwise emit it as a
        # second trace root, sibling to the ``@observe`` span. Stamp the
        # enclosing deepeval span's UUID as a logical-parent override so the
        # exporter can re-parent the OTel root onto it. Only fires for OTel
        # roots; child OTel spans keep their native parent_uuid.
        bridge_otel_root_to_deepeval_parent(
            span, integration=Integration.PYDANTIC_AI.value
        )

        # ----- per-span classification (no settings dependency) -----
        # Span classification (agent / llm / tool) happens at on_start
        # because ``_push_span_context`` reads the assigned
        # ``confident.span.type`` to decide whether to create an
        # ``AgentSpan`` vs a ``BaseSpan`` placeholder. All per-span
        # configuration (metric_collection, metadata, prompt, etc.) is
        # the user's responsibility via ``update_current_span(...)``
        # from inside their tool / agent body — settings deliberately
        # carries no span-level fields.
        operation_name = span.attributes.get("gen_ai.operation.name")
        agent_name = (
            span.attributes.get("gen_ai.agent.name")
            or span.attributes.get("pydantic_ai.agent.name")
            or span.attributes.get("agent_name")
        )

        if agent_name and self._is_agent_span(operation_name):
            self._add_agent_span(span, agent_name)

        if operation_name in self.LLM_OPERATION_NAMES:
            # Explicitly classify model request spans as LLM spans so
            # they're not mislabeled as agent spans when
            # gen_ai.agent.name is present.
            span.set_attribute(ConfidentAttr.SPAN_TYPE, "llm")
        span.set_attribute(
            ConfidentAttr.SPAN_INTEGRATION, Integration.PYDANTIC_AI.value
        )

        # ----- push BaseSpan placeholder so update_current_span works -----
        self._push_span_context(span, agent_name, operation_name)

    def on_end(self, span):
        sid = span.get_span_context().span_id

        # ----- snapshot trace context FRESH at on_end -----
        # Resolved here (not at on_start) so the latest update_current_trace
        # values land on the OTel span. Uses the post-end attr writer because
        # the SDK has already set ``_end_time`` by the time on_end fires,
        # which makes ``span.set_attribute`` a silent no-op.
        try:
            serialize_trace_context_to_otel_attrs(span, self.settings)
        except Exception as exc:
            logger.debug(
                "Failed to serialize trace context for span_id=%s: %s",
                sid,
                exc,
            )

        # ----- pop current_span_context and serialize user mutations -----
        finalize_span_placeholder(span, self._tokens, self._placeholders)

        # ----- catch any agent spans that weren't classified at on_start -----
        already_processed = span.attributes.get(ConfidentAttr.SPAN_TYPE) in {
            "agent",
            "llm",
            "tool",
        }
        if not already_processed:
            operation_name = span.attributes.get("gen_ai.operation.name")
            agent_name = (
                span.attributes.get("gen_ai.agent.name")
                or span.attributes.get("pydantic_ai.agent.name")
                or span.attributes.get("agent_name")
            )
            if agent_name and self._is_agent_span(operation_name):
                self._add_agent_span(span, agent_name)

        attrs = span.attributes or {}
        if not attrs.get(ConfidentAttr.SPAN_INTEGRATION):
            set_span_attribute_post_end(
                span,
                ConfidentAttr.SPAN_INTEGRATION,
                Integration.PYDANTIC_AI.value,
            )
        if attrs.get(ConfidentAttr.SPAN_TYPE) == "llm" and not attrs.get(
            ConfidentAttr.SPAN_PROVIDER
        ):
            model = (
                attrs.get(ConfidentAttr.LLM_MODEL)
                or attrs.get("gen_ai.response.model")
                or attrs.get("gen_ai.request.model")
            )
            provider = infer_provider_from_model(str(model)) if model else None
            if provider:
                provider = normalize_span_provider_for_platform(provider)
                set_span_attribute_post_end(
                    span, ConfidentAttr.SPAN_PROVIDER, provider
                )

        # ----- pop the implicit trace placeholder if we pushed one -----
        # Must run AFTER the trace-context serialization above so that the
        # implicit placeholder's mutations land on this root span's attrs.
        # Only the root span pushed, so only the root span pops; child
        # spans see the placeholder via inherited contextvars but never
        # touch the token.
        pop_implicit_trace_context(
            span, self._trace_tokens, self._trace_placeholders
        )

    def _push_span_context(
        self,
        span,
        agent_name: Optional[str],
        operation_name: Optional[str],
    ) -> None:
        """Create a typed placeholder span and push it onto current_span_context.

        The placeholder is only used as a write target for
        ``update_current_span(...)``. Its fields are serialized back into
        ``confident.span.*`` OTel attributes at ``on_end``. The actual span
        objects shipped to Confident AI are still constructed by the exporter.
        """
        try:
            sid = span.get_span_context().span_id
            tid = span.get_span_context().trace_id
            span_type = span.attributes.get(ConfidentAttr.SPAN_TYPE)
            start_time = (
                peb.epoch_nanos_to_perf_seconds(span.start_time)
                if span.start_time
                else perf_counter()
            )
            kwargs: Dict[str, Any] = dict(
                uuid=to_hex_string(sid, 16),
                trace_uuid=to_hex_string(tid, 32),
                status=TraceSpanStatus.IN_PROGRESS,
                start_time=start_time,
            )
            if span_type == SpanType.AGENT:
                placeholder = AgentSpan(
                    name=(
                        span.attributes.get(ConfidentAttr.SPAN_NAME)
                        or agent_name
                        or "agent"
                    ),
                    **kwargs,
                )
            elif span_type == SpanType.LLM:
                placeholder = LlmSpan(**kwargs)
            else:
                placeholder = BaseSpan(**kwargs)

            # Consume any ``next_*_span(...)`` defaults the user staged
            # for this span. ``pop_pending_for`` returns a one-shot
            # merged dict (base slot + typed slot for ``span_type``) and
            # resets both slots so subsequent spans in the same scope
            # don't re-inherit. ``apply_pending_to_span`` writes the
            # fields onto the placeholder before we push it onto
            # ``current_span_context`` so that any user code that
            # reads the span (or runs ``update_current_span(...)`` later)
            # sees the staged values as the baseline.
            pending = pop_pending_for(span_type)
            if pending:
                apply_pending_to_span(placeholder, pending)

            token = current_span_context.set(placeholder)
            self._tokens[sid] = token
            self._placeholders[sid] = placeholder
        except Exception as exc:
            logger.debug(
                "Failed to push current_span_context placeholder: %s", exc
            )

    def _add_agent_span(self, span, name):
        # Uses the post-end-safe writer because this is called from BOTH
        # ``on_start`` (where set_attribute would also work) and ``on_end``
        # (where it wouldn't, since the SDK has already set ``_end_time``).
        # ``_set_attr_post_end`` writes through the underlying mutable
        # ``_attributes`` mapping in either case.
        set_span_attribute_post_end(span, ConfidentAttr.SPAN_TYPE, "agent")
        set_span_attribute_post_end(span, ConfidentAttr.SPAN_NAME, name)

    def _is_agent_span(self, operation_name: Optional[str]) -> bool:
        return operation_name == "invoke_agent"
