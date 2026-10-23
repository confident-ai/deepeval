import functools
from typing import Callable

from crewai.tools import tool as crewai_tool

from deepeval.tracing.context import update_current_span


def tool(*args, metric=None, metric_collection=None, **kwargs) -> Callable:
    """
    Simple wrapper around crewai.tools.tool that:
      - sets metric and metric_collection on the span the tool runs in
      - accepts additional parameters: metric and metric_collection
      - remains backward compatible with CrewAI's decorator usage patterns
    """
    crewai_kwargs = kwargs

    def build(tool_name: str, f: Callable):
        @functools.wraps(f)
        def wrapped(*f_args, **f_kwargs):
            update_current_span(
                metrics=metric, metric_collection=metric_collection
            )
            return f(*f_args, **f_kwargs)

        return crewai_tool(tool_name, **crewai_kwargs)(wrapped)

    # Case 1: @tool (function passed directly)
    if len(args) == 1 and callable(args[0]):
        return build(args[0].__name__, args[0])

    # Case 2: @tool("name")
    if len(args) == 1 and isinstance(args[0], str):
        return functools.partial(build, args[0])

    # Case 3: @tool(result_as_answer=True, ...) – kwargs only
    if len(args) == 0:
        return lambda f: build(f.__name__, f)

    raise ValueError("Invalid arguments")
