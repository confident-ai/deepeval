import functools
from typing import Callable

from deepeval.tracing.context import current_span_context
from deepeval.tracing.types import ToolSpan
from .subs import try_import_tool_decorator


def deepeval_tool(
    *args, metric=None, metric_collection=None, **kwargs
) -> Callable:
    """
    Simple wrapper around crewai.tools.tool that:
      - prints the original function's input and output
      - accepts additional parameters: metric and metric_collection (unused, for compatibility)
      - remains backward compatible with CrewAI's decorator usage patterns
    """
    crewai_tool = try_import_tool_decorator()
    if crewai_tool is None:
        # return a no-op decorator that preserves function behavior without CrewAI
        # this is just to prevent our decorator from breaking things completely
        def _noop(f):
            return f

        crewai_decorator = _noop
    else:

        def crewai_decorator(name, **kw):
            return crewai_tool(name, **kw)

    crewai_kwargs = kwargs

    # Case 1: @tool (function passed directly)
    if len(args) == 1 and callable(args[0]):
        f = args[0]
        tool_name = f.__name__

        @functools.wraps(f)
        def wrapped(*f_args, **f_kwargs):
            current_span = current_span_context.get()
            if current_span and isinstance(current_span, ToolSpan):
                current_span.metric_collection = metric_collection
                current_span.metrics = metric
            result = f(*f_args, **f_kwargs)
            return result

        return crewai_decorator(tool_name, **crewai_kwargs)(wrapped)

    # Case 2: @tool("name")
    if len(args) == 1 and isinstance(args[0], str):
        tool_name = args[0]

        def _decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapped(*f_args, **f_kwargs):
                current_span = current_span_context.get()
                if current_span and isinstance(current_span, ToolSpan):
                    current_span.metric_collection = metric_collection
                    current_span.metrics = metric
                result = f(*f_args, **f_kwargs)
                return result

            return crewai_decorator(tool_name, **crewai_kwargs)(wrapped)

        return _decorator

    # Case 3: @tool(result_as_answer=True, ...) — kwargs only
    if len(args) == 0:

        def _decorator(f: Callable) -> Callable:
            tool_name = f.__name__

            @functools.wraps(f)
            def wrapped(*f_args, **f_kwargs):
                current_span = current_span_context.get()
                if current_span and isinstance(current_span, ToolSpan):
                    current_span.metric_collection = metric_collection
                    current_span.metrics = metric
                result = f(*f_args, **f_kwargs)
                return result

            return crewai_decorator(tool_name, **crewai_kwargs)(wrapped)

        return _decorator

    raise ValueError("Invalid arguments")
