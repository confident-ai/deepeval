import functools
from typing import Callable
from crewai.tools import tool as crewai_tool

from deepeval.tracing.context import current_span_context
from deepeval.tracing.types import ToolSpan


def tool(*args, metric=None, metric_collection=None, **kwargs) -> Callable:
    """
    Simple wrapper around crewai.tools.tool that:
      - attaches metric and metric_collection as function attributes
      - accepts additional parameters: metric and metric_collection
      - remains backward compatible with CrewAI's decorator usage patterns
    """
    crewai_kwargs = kwargs

    # Case 1: @tool (function passed directly)
    if len(args) == 1 and callable(args[0]):
        f = args[0]
        tool_name = f.__name__

        @functools.wraps(f)
        def wrapped(*f_args, **f_kwargs):
            result = f(*f_args, **f_kwargs)
            return result

        # Attach metrics as attributes to the wrapped function
        # These will be read by the event listener in handler.py
        wrapped._metric_collection = metric_collection
        wrapped._metrics = metric

        # Pass the wrapped function to CrewAI's tool decorator
        tool_instance = crewai_tool(tool_name, **crewai_kwargs)(wrapped)

        # Also attach to the tool instance itself for redundancy
        tool_instance._metric_collection = metric_collection
        tool_instance._metrics = metric

        return tool_instance

    # Case 2: @tool("name")
    if len(args) == 1 and isinstance(args[0], str):
        tool_name = args[0]

        def _decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapped(*f_args, **f_kwargs):
                result = f(*f_args, **f_kwargs)
                return result

            # Attach metrics as attributes
            wrapped._metric_collection = metric_collection
            wrapped._metrics = metric

            tool_instance = crewai_tool(tool_name, **crewai_kwargs)(wrapped)

            # Also attach to the tool instance
            tool_instance._metric_collection = metric_collection
            tool_instance._metrics = metric

            return tool_instance

        return _decorator

    # Case 3: @tool(result_as_answer=True, ...) – kwargs only
    if len(args) == 0:

        def _decorator(f: Callable) -> Callable:
            tool_name = f.__name__

            @functools.wraps(f)
            def wrapped(*f_args, **f_kwargs):
                result = f(*f_args, **f_kwargs)
                return result

            # Attach metrics as attributes
            wrapped._metric_collection = metric_collection
            wrapped._metrics = metric

            tool_instance = crewai_tool(tool_name, **crewai_kwargs)(wrapped)

            # Also attach to the tool instance
            tool_instance._metric_collection = metric_collection
            tool_instance._metrics = metric

            return tool_instance

        return _decorator

    raise ValueError("Invalid arguments")
