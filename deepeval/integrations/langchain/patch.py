import functools
from deepeval.metrics import BaseMetric
from deepeval.tracing.context import update_current_span
from typing import List, Optional, Callable
from langchain_core.tools import tool as original_tool, BaseTool


def tool(
    *args,
    metrics: Optional[List[BaseMetric]] = None,
    metric_collection: Optional[str] = None,
    **kwargs
):
    """
    Patched version of langchain_core.tools.tool that prints inputs and outputs
    """

    # original_tool returns a decorator function, so we need to return a decorator
    def decorator(func: Callable) -> BaseTool:
        func = _patch_tool_decorator(func, metrics, metric_collection)
        tool_instance = original_tool(*args, **kwargs)(func)
        return tool_instance

    return decorator


def _patch_tool_decorator(
    func: Callable,
    metrics: Optional[List[BaseMetric]] = None,
    metric_collection: Optional[str] = None,
):
    original_func = func

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        # update_current_span is a no-op without an active span and
        # skips None values, so metrics staged on the tool span
        # aren't clobbered.
        update_current_span(
            metrics=metrics, metric_collection=metric_collection
        )
        res = original_func(*args, **kwargs)
        return res

    tool = wrapper
    return tool
