import asyncio
from functools import lru_cache
import inspect
from typing import Callable


@lru_cache(maxsize=None)
def _has_kwarg(func: Callable, keyword: str) -> bool:
    """Return True if func accepts keyword or has **kwargs."""
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def _measure_no_indicator(metric, test_case):
    """Call metric.measure(test_case) with _show_indicator=False if supported."""
    measure = getattr(metric, "measure")
    if _has_kwarg(measure, "_show_indicator"):
        return measure(test_case, _show_indicator=False)
    return measure(test_case)


async def _a_measure_no_indicator(metric, test_case):
    """
    Prefer metric.a_measure with fall back to metric.measure in a thread.
    Always disable indicators when supported. This is to prevent interference
    with the gepa indicator.
    """
    a_measure = getattr(metric, "a_measure", None)

    if a_measure is not None:
        call = (
            a_measure(test_case, _show_indicator=False)
            if _has_kwarg(a_measure, "_show_indicator")
            else a_measure(test_case)
        )
        # Be resilient if impl returns a plain value
        return await call if inspect.isawaitable(call) else call

    # No async impl: run sync measure in a thread
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: _measure_no_indicator(metric, test_case)
    )
