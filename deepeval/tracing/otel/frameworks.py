"""Private, optional confident-trace adapters behind existing public entrypoints."""

from __future__ import annotations

import atexit
from threading import RLock

_runtime = None
_enabled = {}
_lock = RLock()
_LABELS = {
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "crewai": "CrewAI",
    "openai_agents": "OpenAI Agents",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}


def available():
    try:
        import wrapt  # noqa: F401 - required by the optional framework emitters
        from confident_trace._core.attachment import instrument_framework
    except ImportError:
        return False
    return callable(instrument_framework)


def enabled(name):
    return name in _enabled


def _get_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    from confident_trace._core import runtime
    from opentelemetry.sdk.trace import TracerProvider
    from deepeval.tracing.otel.provider import configure_owned_sampling

    public = runtime.current()
    if public is not None and public.active:
        _runtime = public
    else:
        provider = TracerProvider(shutdown_on_exit=False)
        configure_owned_sampling(provider)
        _runtime = runtime.Runtime(provider)
    return _runtime


def instrument(name, *, api_key=None, **options):
    """Return false only when the unpublished attachment contract is absent."""
    if not available():
        return False
    with _lock:
        if name in _enabled:
            if api_key is not None:
                _enabled[name][0].reconfigure_api_key(api_key)
            if name == "llamaindex" and options.get("dispatcher") is not None:
                from confident_trace.integrations.llamaindex.instrumentation import (
                    instrument as attach_dispatcher,
                )

                _enabled[name][1].extend(attach_dispatcher(_runtime, **options))
            return True
        from confident_trace._core.attachment import instrument_framework
        from deepeval.integrations.openinference.instrumentator import (
            OpenInferenceInstrumentationSettings,
        )
        from deepeval.tracing.otel.context_aware_processor import (
            ContextAwareSpanProcessor,
        )
        from deepeval.tracing.otel.provider import attach
        from deepeval.tracing.tracing import trace_manager

        if name == "openai_agents":
            try:
                from confident_trace.integrations.openai_agents.instrumentation import (
                    create_processor,  # noqa: F401 - private contract probe
                )
                from openinference.instrumentation.openai_agents import (
                    OpenAIAgentsInstrumentor,  # noqa: F401 - optional extra probe
                )
            except ImportError:
                return False

        rt = _get_runtime()
        settings = OpenInferenceInstrumentationSettings(
            integration=_LABELS[name]
        )
        processor = ContextAwareSpanProcessor(
            api_key or trace_manager.confident_api_key
        )
        interceptor = _interceptor(settings)
        router = attach(rt.provider, name, interceptor, processor)
        # The private runtime owns no exporter; this gate is only for native scope
        # stamping required by the OpenAI Agents adapter.
        if rt.processor is None:
            rt.processor = router.native_attachment
        try:
            undo = instrument_framework(rt, name, **options)
        except BaseException:
            router.adapters.pop(name, None)
            processor.shutdown()
            raise
        _enabled[name] = (processor, undo)
        return True


def _interceptor(settings):
    from deepeval.integrations.openinference.instrumentator import (
        OpenInferenceSpanInterceptor,
    )

    return OpenInferenceSpanInterceptor(settings)


def reset(name):
    with _lock:
        entry = _enabled.pop(name, None)
        if entry is not None:
            try:
                for undo in reversed(entry[1]):
                    undo()
            finally:
                from deepeval.tracing.otel.provider import _providers

                router = _providers.get(_runtime.provider) if _runtime else None
                if router is not None:
                    router.adapters.pop(name, None)
                entry[0].shutdown()


def shutdown():
    for name in tuple(_enabled):
        reset(name)


atexit.register(shutdown)


def bind_langchain(handler):
    """Forward explicit callbacks to the shared emitter; do not auto-enroll runs."""
    if not instrument("langchain", auto_register=False):
        return
    from deepeval.tracing.context import (
        current_trace_context,
        current_span_context,
    )

    bridge = _get_runtime()._langchain_bridge
    handler.run_inline = True

    def callback(method):
        def invoke(*args, **kwargs):
            if (
                method == "on_chain_start"
                and kwargs.get("parent_run_id") is None
            ):
                kwargs = {**kwargs, "run_type": "agent"}
            result = getattr(bridge, method)(*args, **kwargs)
            if method.endswith("_start"):
                live = current_span_context.get()
                trace = current_trace_context.get()
                if trace is not None and kwargs.get("parent_run_id") is None:
                    for field, value in handler._original_init_fields.items():
                        if value is not None:
                            setattr(trace, field, value)
                    handler._trace = trace
                    handler.trace_uuid = trace.uuid
                    if method == "on_chain_start":
                        if handler.metrics is not None:
                            trace.metrics = handler.metrics
                        if handler.metric_collection is not None:
                            trace.metric_collection = handler.metric_collection
                if live is not None:
                    from deepeval.tracing.types import LlmSpan, RetrieverSpan
                    from deepeval.tracing.context import apply_pending_to_span

                    md = kwargs.get("metadata") or {}
                    fields = (
                        ("metrics", "metric_collection", "prompt")
                        if isinstance(live, LlmSpan)
                        else (
                            ("metric_collection",)
                            if isinstance(live, RetrieverSpan)
                            else ()
                        )
                    )
                    apply_pending_to_span(
                        live,
                        {
                            field: md[field]
                            for field in fields
                            if field in md
                            and field not in live.model_fields_set
                        },
                    )
                    if isinstance(live, RetrieverSpan) and not live.embedder:
                        live.embedder = md.get(
                            "ls_embedding_provider", "unknown"
                        )
            return result

        return invoke

    for method in dir(bridge):
        if method.startswith("on_") and callable(getattr(bridge, method)):
            setattr(handler, method, callback(method))
    handler.on_chat_model_end = callback("on_llm_end")
    handler.on_chat_model_error = callback("on_llm_error")


def bind_crewai_metrics():
    """Copy wrapper-class metric options while the emitted span is live."""
    from deepeval.tracing.context import current_span_context

    def apply(kind, instance, operation):
        live = current_span_context.get()
        if live is not None:
            for field in ("metrics", "metric_collection"):
                value = getattr(instance, "_" + field, None)
                if value is not None:
                    setattr(live, field, value)

    state = _get_runtime()._crewai_state
    previous = getattr(state, "on_operation_start", None)
    state.on_operation_start = apply

    def restore():
        if state.on_operation_start is apply:
            state.on_operation_start = previous

    _enabled["crewai"][1].append(restore)


def reset_crewai_state():
    if enabled("crewai"):
        api_key = _enabled["crewai"][0]._api_key
        reset("crewai")
        instrument("crewai", api_key=api_key, include_llm=True)
        bind_crewai_metrics()


def flush(name):
    entry = _enabled.get(name)
    return entry[0].force_flush() if entry is not None else True
