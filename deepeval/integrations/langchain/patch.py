from deepeval.metrics import BaseMetric
from typing import List, Optional, Callable
from langchain_core.tools import tool as original_tool, BaseTool

# Collision-safe namespace key for deepeval metadata on tool instances
DEEPEVAL_TOOL_METADATA_KEY = "deepeval"


def tool(
    *args,
    metrics: Optional[List[BaseMetric]] = None,
    metric_collection: Optional[str] = None,
    **kwargs
):
    """
    Patched version of langchain_core.tools.tool that stores deepeval metadata
    on the tool instance for retrieval in the callback handler.

    The metric_collection and metrics are stored under tool_instance.metadata["deepeval"]
    to avoid collision with user metadata.
    """

    def decorator(func: Callable) -> BaseTool:
        tool_instance = original_tool(*args, **kwargs)(func)

        # Store deepeval-specific overrides in a collision-safe namespace
        # The callback handler will check this to override inherited values
        if metrics is not None or metric_collection is not None:
            if tool_instance.metadata is None:
                tool_instance.metadata = {}
            tool_instance.metadata[DEEPEVAL_TOOL_METADATA_KEY] = {
                "metric_collection": metric_collection,
                "metrics": metrics,
            }

        return tool_instance

    return decorator
