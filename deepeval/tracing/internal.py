import functools
import inspect
import sys
from contextlib import nullcontext
from typing import Callable, ContextManager, List, Optional


def observe_methods(
    cls,
    span_type: Optional[str] = None,
    allowed_methods: Optional[List[str]] = None,
    suppress_confident_trace: bool = False,
):
    from deepeval.tracing.tracing import observe

    is_traceable = lambda v: inspect.isfunction(
        v
    ) or inspect.iscoroutinefunction(v)

    methods = {
        k: v
        for k, v in cls.__dict__.items()
        if not k.startswith("__") and is_traceable(v)
    }

    if allowed_methods is not None:
        methods = {k: v for k, v in methods.items() if k in allowed_methods}

    for name, method in methods.items():
        if getattr(method, "_is_deepeval_observed", False):
            continue
        if suppress_confident_trace:
            method = _without_confident_trace(method)
        setattr(
            cls,
            name,
            observe(
                type=span_type,
                _drop_if_root=True,
                _internal=True,
            )(method),
        )


def _without_confident_trace(method: Callable) -> Callable:
    """Keep what ``method`` calls, such as a judge model, out of user traces."""
    if inspect.isgeneratorfunction(method) or inspect.isasyncgenfunction(
        method
    ):
        # A scope can't span a generator's lazy execution.
        return method

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def asynchronous(*args, **kwargs):
            with _confident_trace_suppressed():
                return await method(*args, **kwargs)

        return asynchronous

    @functools.wraps(method)
    def synchronous(*args, **kwargs):
        with _confident_trace_suppressed():
            return method(*args, **kwargs)

    return synchronous


def _confident_trace_suppressed() -> ContextManager:
    # Only deepeval's setup module starts confident-trace, so when it was never
    # imported there is nothing to suppress, and importing it would load OTel.
    setup = sys.modules.get("deepeval.tracing.otel.setup")
    return setup.suppress_confident_trace() if setup else nullcontext()
