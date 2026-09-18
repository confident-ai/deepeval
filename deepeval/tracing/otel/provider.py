"""One DeepEval pipeline per provider; application exporters remain independent."""

from __future__ import annotations

from threading import RLock
from weakref import WeakKeyDictionary

from opentelemetry.sdk.trace import SpanProcessor
from deepeval.tracing.context import current_trace_context

_providers = WeakKeyDictionary()
_lock = RLock()


class IntegrationRouter(SpanProcessor):
    def __init__(self):
        self.adapters = {}
        self.active = {}
        self.lock = RLock()
        self.capture = None

    def register(self, name, interceptor, processor):
        with self.lock:
            old = self.adapters.get(name)
            capture = getattr(processor, "_capture", None)
            if capture is not None:
                if self.capture is None:
                    self.capture = capture
                else:
                    processor._capture = self.capture
            self.adapters[name] = (interceptor, processor)
            return old

    def on_start(self, span, parent_context=None):
        with self.lock:
            if not self.adapters:
                return
            from deepeval.tracing.context import prune_otel_context
            from deepeval.tracing.otel.capture import current_eval_owner
            from deepeval.tracing.tracing import trace_manager

            session = current_eval_owner.get()
            if (
                session is not None
                and session is not trace_manager.eval_session
            ):
                return
            prune_otel_context()
            context = span.get_span_context()
            key = (context.trace_id, context.span_id)
            parent = (
                (context.trace_id, span.parent.span_id) if span.parent else None
            )
            adapter = self.active.get(parent)
            scope = getattr(span.instrumentation_scope, "name", "") or ""
            integration = (span.attributes or {}).get(
                "confident.span.integration", ""
            )
            if "pydantic" in scope or integration == "PydanticAI":
                name = "pydantic_ai"
            elif "strands" in scope or integration == "Strands":
                name = "strands"
            elif (
                "adk" in scope
                or scope == "gcp.vertex.agent"
                or integration == "Google ADK"
            ):
                name = "openinference"
            elif "agentcore" in scope or integration == "AgentCore":
                name = "agentcore"
            else:
                from deepeval.tracing.otel.frameworks import _LABELS

                name = next(
                    (
                        key
                        for key, value in _LABELS.items()
                        if value == integration
                    ),
                    "openai_agents"
                    if "openai_agents" in scope
                    else ("openinference" if adapter is None else None),
                )
            adapter = (
                self.adapters.get(name)
                or adapter
                or next(iter(self.adapters.values()))
            )
            self.active[key] = adapter
            interceptor, processor = adapter
            interceptor.on_start(span, parent_context)
            processor.on_start(span, parent_context)
            capture = getattr(processor, "_capture", None)
            binding = capture.bindings.get(key) if capture is not None else None
            if binding is not None:
                binding.cleanup = lambda: self._forget(key, interceptor)

    def _forget(self, key, interceptor):
        self.active.pop(key, None)
        interceptor.discard_context(key)

    def on_end(self, span):
        with self.lock:
            context = span.get_span_context()
            adapter = self.active.pop((context.trace_id, context.span_id), None)
            if adapter is None:
                return
            interceptor, processor = adapter
            capture = getattr(processor, "_capture", None)
            binding = (
                capture.bindings.get((context.trace_id, context.span_id))
                if capture is not None
                else None
            )
            # End may run after detachment or on a worker. Serialize the original
            # mutable trace context, never another golden's current context.
            token = (
                current_trace_context.set(binding.trace_context)
                if binding
                and binding.trace_context is not None
                and binding.trace_context is not current_trace_context.get()
                else None
            )
            try:
                if binding is not None and binding.placeholder is not None:
                    from deepeval.tracing import perf_epoch_bridge as peb

                    binding.placeholder.end_time = (
                        peb.epoch_nanos_to_perf_seconds(span.end_time)
                    )
                interceptor.on_end(span)
                processor.on_end(span)
            finally:
                if token is not None:
                    current_trace_context.reset(token)

    def force_flush(self, timeout_millis=30000):
        with self.lock:
            adapters = list(self.adapters.values())
        results = [p.force_flush(timeout_millis) for _, p in adapters]
        return all(results)

    def shutdown(self):
        with self.lock:
            adapters = list(self.adapters.values())
        for _, processor in adapters:
            processor.shutdown()


def attach(provider, name, interceptor, processor):
    with _lock:
        router = _providers.get(provider)
        if router is None:
            router = IntegrationRouter()
            _providers[provider] = router
            try:
                from confident_trace._core.attachment import attach_native
            except ImportError:
                # Python 3.9 and installations without the unpublished SDK
                # retain direct OTEL attachment. No network/package install here.
                provider.add_span_processor(router)
                router.native_attachment = None
            else:
                router.native_attachment = attach_native(provider, router)
        if router.native_attachment is not None and name in {
            "pydantic_ai",
            "strands",
            "agentcore",
            "openinference",
        }:
            native_name = "google_adk" if name == "openinference" else name
            router.native_attachment.enable((native_name,))
        router.register(name, interceptor, processor)
    return router


def configure_owned_sampling(provider):
    """Record evaluation spans while preserving production sampling decisions.

    Only call immediately after DeepEval creates a provider. Never change the
    sampler on a provider supplied by the application.
    """
    from opentelemetry.sdk.trace.sampling import Sampler, ALWAYS_ON

    delegate = provider.sampler

    class EvaluationSampler(Sampler):
        def should_sample(self, *args, **kwargs):
            from deepeval.tracing.otel.capture import current_eval_owner
            from deepeval.contextvars import get_current_golden
            from deepeval.tracing.tracing import trace_manager

            session = current_eval_owner.get()
            evaluating = (
                session is trace_manager.eval_session and session is not None
            ) or (
                trace_manager.is_evaluating and get_current_golden() is not None
            )
            sampler = ALWAYS_ON if evaluating else delegate
            return sampler.should_sample(*args, **kwargs)

        def get_description(self):
            return f"DeepEvalEvaluation({delegate.get_description()})"

    provider.sampler = EvaluationSampler()
