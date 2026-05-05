from __future__ import annotations

import contextvars
import json
import logging
import warnings
from time import perf_counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.context import (
    apply_pending_to_span,
    current_span_context,
    current_trace_context,
    pop_pending_for,
)
from deepeval.tracing.otel.context_aware_processor import (
    ContextAwareSpanProcessor,
)
from deepeval.tracing.otel.utils import (
    stash_pending_metrics,
    to_hex_string,
)
from deepeval.tracing.perf_epoch_bridge import init_clock_bridge
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    Trace,
    TraceSpanStatus,
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
        # ``_serialize_placeholder_to_otel_attrs``.

        # ----- push implicit trace context for bare agent.run callers -----
        # If the caller didn't wrap in ``@observe`` / ``with trace(...)`` and
        # this is the OTel root span, push an implicit ``Trace`` placeholder
        # onto ``current_trace_context`` so ``update_current_trace(...)``
        # from inside tools / nested helpers actually mutates something.
        # The placeholder is tagged ``is_otel_implicit=True`` so that
        # ``ContextAwareSpanProcessor`` keeps routing to OTLP (caller didn't
        # opt into REST). Mutations are picked up automatically by the
        # existing per-span ``_serialize_trace_context_to_otel_attrs`` since
        # it reads from ``current_trace_context`` at every ``on_end``.
        self._maybe_push_implicit_trace_context(span)

        # ----- bridge OTel root span to enclosing deepeval span -----
        # When an OTel root span starts inside a deepeval-managed span (the
        # canonical case being ``@observe(type="agent") -> agent.run(...)``),
        # OTel sees no parent and the exporter would otherwise emit it as a
        # second trace root, sibling to the ``@observe`` span. Stamp the
        # enclosing deepeval span's UUID as a logical-parent override so the
        # exporter can re-parent the OTel root onto it. Only fires for OTel
        # roots; child OTel spans keep their native parent_uuid.
        self._maybe_bridge_otel_root_to_deepeval_parent(span)

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
            span.set_attribute("confident.span.type", "llm")

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
            self._serialize_trace_context_to_otel_attrs(span)
        except Exception as exc:
            logger.debug(
                "Failed to serialize trace context for span_id=%s: %s",
                sid,
                exc,
            )

        # ----- pop current_span_context and serialize user mutations -----
        placeholder = self._placeholders.pop(sid, None)
        token = self._tokens.pop(sid, None)
        if token is not None:
            try:
                current_span_context.reset(token)
            except Exception as exc:
                logger.debug(
                    "Failed to reset current_span_context for span_id=%s: %s",
                    sid,
                    exc,
                )
        if placeholder is not None:
            try:
                self._serialize_placeholder_to_otel_attrs(placeholder, span)
            except Exception as exc:
                logger.debug(
                    "Failed to serialize span placeholder for span_id=%s: %s",
                    sid,
                    exc,
                )
            # ``BaseMetric`` instances can't ride in OTel attrs (primitives
            # only), so hand them to the in-process overlay for the exporter
            # to re-attach. Eval-mode gate prevents the registry from growing
            # in prod paths where the OTLP collector lives in another process
            # and the reader never fires.
            try:
                if placeholder.metrics and trace_manager.is_evaluating:
                    stash_pending_metrics(
                        to_hex_string(sid, 16), placeholder.metrics
                    )
            except Exception as exc:
                logger.debug(
                    "Failed to stash pending metrics for span_id=%s: %s",
                    sid,
                    exc,
                )

        # ----- catch any agent spans that weren't classified at on_start -----
        already_processed = span.attributes.get("confident.span.type") in {
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

        # ----- pop the implicit trace placeholder if we pushed one -----
        # Must run AFTER the trace-context serialization above so that the
        # implicit placeholder's mutations land on this root span's attrs.
        # Only the root span pushed, so only the root span pops; child
        # spans see the placeholder via inherited contextvars but never
        # touch the token.
        self._maybe_pop_implicit_trace_context(span)

    def _push_span_context(
        self,
        span,
        agent_name: Optional[str],
        operation_name: Optional[str],
    ) -> None:
        """Create a placeholder BaseSpan and push it onto current_span_context.

        The placeholder is only used as a write target for
        ``update_current_span(...)``. Its fields are serialized back into
        ``confident.span.*`` OTel attributes at ``on_end``. The actual span
        objects shipped to Confident AI are still constructed by the exporter.
        """
        try:
            sid = span.get_span_context().span_id
            tid = span.get_span_context().trace_id
            span_type = span.attributes.get("confident.span.type")
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
            if span_type == "agent":
                placeholder = AgentSpan(
                    name=(
                        span.attributes.get("confident.span.name")
                        or agent_name
                        or "agent"
                    ),
                    **kwargs,
                )
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

    def _maybe_push_implicit_trace_context(self, span) -> None:
        """Push an implicit ``Trace`` placeholder for bare ``agent.run`` callers.

        Symmetric to ``_push_span_context``, but at the trace level. Only
        fires for the OTel root span AND only when the caller hasn't
        already pushed their own trace context (via ``@observe`` / ``with
        trace(...)``). The placeholder exists solely so that
        ``update_current_trace(...)`` from inside tools / nested helpers
        has a target to mutate; mutations are picked up automatically by
        the existing per-span ``_serialize_trace_context_to_otel_attrs``.

        Tagged ``is_otel_implicit=True`` so ``ContextAwareSpanProcessor``
        knows NOT to switch routing to REST — bare callers expect OTLP.
        """
        if current_trace_context.get() is not None:
            return  # user already owns the trace context; don't touch it
        # Only the OTel root span pushes; child spans inherit the placeholder
        # via contextvars and never need their own.
        if getattr(span, "parent", None) is not None:
            return
        try:
            sid = span.get_span_context().span_id
            tid = span.get_span_context().trace_id
            start_time = (
                peb.epoch_nanos_to_perf_seconds(span.start_time)
                if span.start_time
                else perf_counter()
            )
            implicit = Trace(
                uuid=to_hex_string(tid, 32),
                root_spans=[],
                status=TraceSpanStatus.IN_PROGRESS,
                start_time=start_time,
                is_otel_implicit=True,
            )
            token = current_trace_context.set(implicit)
            self._trace_tokens[sid] = token
            self._trace_placeholders[sid] = implicit
        except Exception as exc:
            logger.debug(
                "Failed to push implicit current_trace_context: %s", exc
            )

    def _maybe_bridge_otel_root_to_deepeval_parent(self, span) -> None:
        """Re-parent an OTel root span onto its enclosing deepeval span.

        When ``@observe(type="agent")`` (or any deepeval-managed span) wraps
        a bare ``agent.run(...)`` call, the deepeval span is created off-OTel
        and pushed onto ``current_span_context``, but no OTel parent context
        is established. Pydantic AI then opens an OTel root span (no native
        parent), and the exporter would otherwise emit it as a second trace
        root sibling to the ``@observe`` span — visually the two appear as
        two separate agent spans rather than parent → child.

        We close that gap by stamping the deepeval span's UUID onto the OTel
        root as ``confident.span.parent_uuid``. ``ConfidentSpanExporter``
        prefers this override iff the OTel span has no native parent, so the
        re-parenting only affects the dual-root case and never overrides a
        real OTel parent_id for nested OTel spans.
        """
        # Only OTel roots need bridging; child OTel spans already have a
        # real parent_id pointing into the same OTel trace.
        if getattr(span, "parent", None) is not None:
            return
        parent_span = current_span_context.get()
        if parent_span is None:
            return
        parent_uuid = getattr(parent_span, "uuid", None)
        if not parent_uuid:
            return
        try:
            self._set_attr_post_end(
                span, "confident.span.parent_uuid", parent_uuid
            )
        except Exception as exc:
            logger.debug(
                "Failed to bridge OTel root span to deepeval parent "
                "(parent_uuid=%s): %s",
                parent_uuid,
                exc,
            )

    def _maybe_pop_implicit_trace_context(self, span) -> None:
        """Pop the implicit trace placeholder pushed at ``on_start``.

        No-op for spans that didn't push (children, or roots that found a
        user-owned context already in place).
        """
        try:
            sid = span.get_span_context().span_id
        except Exception:
            return
        token = self._trace_tokens.pop(sid, None)
        self._trace_placeholders.pop(sid, None)
        if token is None:
            return
        try:
            current_trace_context.reset(token)
        except Exception as exc:
            logger.debug(
                "Failed to reset implicit current_trace_context for "
                "span_id=%s: %s",
                sid,
                exc,
            )

    @staticmethod
    def _set_attr_post_end(span, key: str, value: Any) -> None:
        """Write an attribute onto a span that may already have ended.

        ``Span.set_attribute`` becomes a silent no-op once ``Span.end()`` has
        been called (the SDK guards on ``self._end_time is not None`` and just
        logs a warning), and the SDK invokes ``on_end`` AFTER setting
        ``_end_time`` — so the obvious ``span.set_attribute(...)`` from inside
        ``SpanInterceptor.on_end`` never lands.

        However the live span constructs its ``_attributes`` as a
        ``BoundedAttributes`` with ``immutable=False`` and passes that same
        dict by reference into ``_readable_span()`` (the ReadableSpan passed to
        all processors). Writing through the mapping's ``__setitem__``
        bypasses the ended-span guard while still respecting the bounded-size
        limits. SpanProcessors fire in registration order, so writes from
        ``SpanInterceptor.on_end`` are visible to ``ConfidentSpanExporter``
        downstream.

        We fall back to ``span.set_attribute`` if the private API ever
        disappears — that path will warn-and-drop, but at least it won't
        crash.
        """
        try:
            attrs = getattr(span, "_attributes", None)
            if attrs is not None:
                attrs[key] = value
                return
        except Exception as exc:
            logger.debug(
                "Direct _attributes write failed for %s; "
                "falling back to set_attribute (may be dropped): %s",
                key,
                exc,
            )
        try:
            span.set_attribute(key, value)
        except Exception as exc:
            logger.debug("set_attribute fallback failed for %s: %s", key, exc)

    @classmethod
    def _serialize_placeholder_to_otel_attrs(
        cls, placeholder: BaseSpan, span
    ) -> None:
        """Mirror update_current_span writes onto confident.span.* attrs.

        Only writes attrs the user actively set on the placeholder. Existing
        attrs already populated by ``on_start`` (e.g. ``confident.span.name``
        when the agent name was discovered, or ``confident.span.metric_collection``
        from settings) are not overwritten by empty placeholder fields.
        """
        if placeholder.metadata:
            cls._set_attr_post_end(
                span,
                "confident.span.metadata",
                json.dumps(placeholder.metadata, default=str),
            )
        if placeholder.input is not None:
            cls._set_attr_post_end(
                span,
                "confident.span.input",
                json.dumps(placeholder.input, default=str),
            )
        if placeholder.output is not None:
            cls._set_attr_post_end(
                span,
                "confident.span.output",
                json.dumps(placeholder.output, default=str),
            )
        if placeholder.metric_collection:
            cls._set_attr_post_end(
                span,
                "confident.span.metric_collection",
                placeholder.metric_collection,
            )
        if placeholder.retrieval_context:
            cls._set_attr_post_end(
                span,
                "confident.span.retrieval_context",
                json.dumps(placeholder.retrieval_context),
            )
        if placeholder.context:
            cls._set_attr_post_end(
                span,
                "confident.span.context",
                json.dumps(placeholder.context),
            )
        if placeholder.expected_output:
            cls._set_attr_post_end(
                span,
                "confident.span.expected_output",
                placeholder.expected_output,
            )
        if placeholder.name and not span.attributes.get("confident.span.name"):
            cls._set_attr_post_end(
                span, "confident.span.name", placeholder.name
            )

    def _serialize_trace_context_to_otel_attrs(self, span) -> None:
        """Resolve trace-level attrs FRESH and write to ``confident.trace.*``.

        Reads from ``current_trace_context`` (so ``update_current_trace(...)``
        from anywhere in the call stack lands on every OTel span) with
        ``DeepEvalInstrumentationSettings`` trace defaults (``name``,
        ``thread_id``, ``user_id``, ``tags``, ``metadata``,
        ``metric_collection``, ``test_case_id``, ``turn_id``) as
        fallback. Metadata merges settings as base + runtime context on
        top.

        Called at ``on_end`` (not ``on_start``) so the latest values are
        captured rather than a stale snapshot. Goes through
        ``_set_attr_post_end`` so it works after the SDK has finalized the
        span's ``_end_time``.
        """
        trace_ctx = current_trace_context.get()

        _name = (trace_ctx.name if trace_ctx else None) or self.settings.name
        _thread_id = (
            trace_ctx.thread_id if trace_ctx else None
        ) or self.settings.thread_id
        _user_id = (
            trace_ctx.user_id if trace_ctx else None
        ) or self.settings.user_id
        _tags = (trace_ctx.tags if trace_ctx else None) or self.settings.tags
        _test_case_id = (
            trace_ctx.test_case_id if trace_ctx else None
        ) or self.settings.test_case_id
        _turn_id = (
            trace_ctx.turn_id if trace_ctx else None
        ) or self.settings.turn_id
        _trace_metric_collection = (
            trace_ctx.metric_collection if trace_ctx else None
        ) or self.settings.metric_collection
        _metadata = {
            **(self.settings.metadata or {}),
            **((trace_ctx.metadata or {}) if trace_ctx else {}),
        }

        if _name:
            self._set_attr_post_end(span, "confident.trace.name", _name)
        if _thread_id:
            self._set_attr_post_end(
                span, "confident.trace.thread_id", _thread_id
            )
        if _user_id:
            self._set_attr_post_end(span, "confident.trace.user_id", _user_id)
        if _tags:
            self._set_attr_post_end(span, "confident.trace.tags", _tags)
        if _metadata:
            self._set_attr_post_end(
                span, "confident.trace.metadata", json.dumps(_metadata)
            )
        if _trace_metric_collection:
            self._set_attr_post_end(
                span,
                "confident.trace.metric_collection",
                _trace_metric_collection,
            )
        if _test_case_id:
            self._set_attr_post_end(
                span, "confident.trace.test_case_id", _test_case_id
            )
        if _turn_id:
            self._set_attr_post_end(span, "confident.trace.turn_id", _turn_id)
        if self.settings.environment:
            self._set_attr_post_end(
                span,
                "confident.trace.environment",
                self.settings.environment,
            )

    def _add_agent_span(self, span, name):
        # Uses the post-end-safe writer because this is called from BOTH
        # ``on_start`` (where set_attribute would also work) and ``on_end``
        # (where it wouldn't, since the SDK has already set ``_end_time``).
        # ``_set_attr_post_end`` writes through the underlying mutable
        # ``_attributes`` mapping in either case.
        self._set_attr_post_end(span, "confident.span.type", "agent")
        self._set_attr_post_end(span, "confident.span.name", name)

    def _is_agent_span(self, operation_name: Optional[str]) -> bool:
        return operation_name == "invoke_agent"
