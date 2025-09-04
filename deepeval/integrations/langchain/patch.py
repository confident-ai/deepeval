from langchain_core.tools import tool as original_tool, BaseTool
from deepeval.metrics import BaseMetric
from typing import List, Optional, Callable, Any
from functools import wraps


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

        # Apply the original tool decorator to get the BaseTool
        tool_instance = original_tool(*args, **kwargs)(func)

        if isinstance(tool_instance, BaseTool):
            if tool_instance.metadata is None:
                tool_instance.metadata = {}

            tool_instance.metadata["metric_collection"] = metric_collection
            tool_instance.metadata["metrics"] = metrics

        return tool_instance

    return decorator
