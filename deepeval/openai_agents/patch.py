"""Let a deepeval ``Agent`` and ``function_tool`` configure the spans
OpenInference opens for them."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace as otel_trace

from deepeval.metrics import BaseMetric
from deepeval.prompt import Prompt
from deepeval.tracing.context import (
    staged_next_span,
    update_agent_span,
    update_current_span,
)

try:
    from agents import function_tool as _agents_function_tool  # type: ignore
    from agents.models.interface import Model
except ImportError:
    _agents_function_tool = None
    Model = object

# The OpenTelemetry span open for each agent span, by the SDK's span id.
agent_spans: Dict[str, otel_trace.Span] = {}

_RUN_STEPS = ("run_single_turn", "run_single_turn_streamed")


def _resolve_agent_from_run_step_args(args: Any, kwargs: Any) -> Any:
    agent = kwargs.get("agent")
    if agent is not None:
        return agent
    bindings = kwargs.get("bindings")
    if bindings is not None:
        public = getattr(bindings, "public_agent", None)
        if public is not None:
            return public
        return getattr(bindings, "execution_agent", None)
    if len(args) > 0:
        return args[0]
    return None


def _with_tool_metrics(
    func: Callable[..., Any],
    metrics: Optional[List[BaseMetric]] = None,
    metric_collection: Optional[str] = None,
) -> Callable[..., Any]:
    """``func``, setting the metrics of the tool span it runs in."""
    if getattr(func, "_is_deepeval_observed", False):
        return func

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def observed(*args: Any, **kwargs: Any) -> Any:
            update_current_span(
                metrics=metrics, metric_collection=metric_collection
            )
            return await func(*args, **kwargs)

    else:

        @wraps(func)
        def observed(*args: Any, **kwargs: Any) -> Any:
            update_current_span(
                metrics=metrics, metric_collection=metric_collection
            )
            return func(*args, **kwargs)

    setattr(observed, "_is_deepeval_observed", True)
    try:
        observed.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    except Exception:
        pass
    return observed


def function_tool(
    func: Optional[Callable[..., Any]] = None, /, *args: Any, **kwargs: Any
) -> Any:
    metrics = kwargs.pop("metrics", None)
    metric_collection = kwargs.pop("metric_collection", None)

    if _agents_function_tool is None:
        raise RuntimeError(
            "agents.function_tool is not available. Please install agents via your package manager"
        )

    if callable(func):
        wrapped = _with_tool_metrics(
            func,
            metrics=metrics,
            metric_collection=metric_collection,
        )
        return _agents_function_tool(wrapped, *args, **kwargs)

    def decorator(real_func: Callable[..., Any]) -> Any:
        wrapped = _with_tool_metrics(
            real_func,
            metrics=metrics,
            metric_collection=metric_collection,
        )
        return _agents_function_tool(wrapped, *args, **kwargs)

    return decorator


class _ObservedModel(Model):
    """Stages an agent's LLM settings for each model call's span."""

    def __init__(
        self,
        inner: Model,
        llm_metric_collection: Optional[str] = None,
        llm_metrics: Optional[List[BaseMetric]] = None,
        confident_prompt: Optional[Prompt] = None,
    ) -> None:
        self._inner = inner
        self._span_fields = {
            "metrics": llm_metrics,
            "metric_collection": llm_metric_collection,
            "prompt": confident_prompt,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def get_response(self, *args, **kwargs):
        with staged_next_span("llm", self._span_fields):
            return await self._inner.get_response(*args, **kwargs)

    def stream_response(self, *args, **kwargs):
        async def stream():
            with staged_next_span("llm", self._span_fields):
                async for event in self._inner.stream_response(*args, **kwargs):
                    yield event

        return stream()


def patch_run_steps() -> None:
    """Give each agent span its agent's settings once a turn has run."""
    import agents.run as agents_run
    import agents.run_internal.run_loop as run_loop

    for name in _RUN_STEPS:
        original = getattr(run_loop, name, None)
        if original is None or getattr(original, "_deepeval_patched", False):
            continue

        def patched_step(original):
            @wraps(original)
            async def patched(*args, **kwargs):
                result = await original(*args, **kwargs)
                try:
                    _configure_agent_span(
                        kwargs.get("agent_span"),
                        _resolve_agent_from_run_step_args(args, kwargs),
                    )
                except Exception:
                    # Never break the run over its tracing.
                    pass
                return result

            patched._deepeval_patched = True
            return patched

        patched = patched_step(original)
        setattr(run_loop, name, patched)
        if getattr(agents_run, name, None) is original:
            setattr(agents_run, name, patched)


def patch_default_agent_runner_get_model():
    try:
        import agents.run_internal.run_loop as run_loop
    except ImportError:
        return

    # Depending on the exact minor version, it might be public or private
    if hasattr(run_loop, "get_model"):
        target_func_name = "get_model"
    elif hasattr(run_loop, "_get_model"):
        target_func_name = "_get_model"
    else:
        return

    original_get_model = getattr(run_loop, target_func_name)
    if getattr(original_get_model, "_deepeval_patched", False):
        return

    def patched_get_model(*args, **kwargs) -> Model:
        model = original_get_model(*args, **kwargs)

        agent = (
            kwargs.get("agent")
            if "agent" in kwargs
            else (args[0] if args else None)
        )
        if agent is None or isinstance(model, _ObservedModel):
            return model

        return _ObservedModel(
            inner=model,
            llm_metric_collection=getattr(agent, "llm_metric_collection", None),
            llm_metrics=getattr(agent, "llm_metrics", None),
            confident_prompt=getattr(agent, "confident_prompt", None),
        )

    patched_get_model.__name__ = original_get_model.__name__
    patched_get_model.__doc__ = original_get_model.__doc__
    patched_get_model._deepeval_patched = True
    setattr(run_loop, target_func_name, patched_get_model)


def _configure_agent_span(agent_span: Any, agent: Any) -> None:
    otel_span = (
        agent_spans.get(agent_span.span_id) if agent_span is not None else None
    )
    if otel_span is None:
        return
    with otel_trace.use_span(otel_span, end_on_exit=False):
        update_current_span(
            metrics=getattr(agent, "agent_metrics", None),
            metric_collection=getattr(agent, "agent_metric_collection", None),
        )
        update_agent_span(
            available_tools=agent_span.span_data.tools,
            agent_handoffs=agent_span.span_data.handoffs,
        )
