from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from deepeval.tracing import observe

try:
    from agents import function_tool as _agents_function_tool  # type: ignore
except Exception:
    _agents_function_tool = None  # type: ignore


def _wrap_with_observe(
    func: Callable[..., Any],
    metrics: Optional[str] = None,
    metric_collection: Optional[str] = None,
) -> Callable[..., Any]:
    # Avoid double-wrapping if already observed
    if getattr(func, "_is_deepeval_observed", False):
        return func

    observed = observe(
        metrics=metrics,
        metric_collection=metric_collection,
        type="tool",
    )(func)

    # Preserve the original signature so the tool schema generation remains correct
    try:
        observed.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    except Exception:
        pass
    return observed


def function_tool(func: Optional[Callable[..., Any]] = None, /, *args: Any, **kwargs: Any) -> Any:
    metrics = kwargs.pop("metrics", None)
    metric_collection = kwargs.pop("metric_collection", None)

    if _agents_function_tool is None:
        raise RuntimeError("agents.function_tool is not available. Please install agents via your package manager")

    if callable(func):
        wrapped = _wrap_with_observe(func, metrics=metrics, metric_collection=metric_collection)
        return _agents_function_tool(wrapped, *args, **kwargs)

    def decorator(real_func: Callable[..., Any]) -> Any:
        wrapped = _wrap_with_observe(real_func, metrics=metrics, metric_collection=metric_collection)
        return _agents_function_tool(wrapped, *args, **kwargs)

    return decorator