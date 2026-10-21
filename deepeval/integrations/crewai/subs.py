import functools
import inspect
from typing import Any, Callable, List, Optional, Type, TypeVar

from pydantic import PrivateAttr

from deepeval.metrics.base_metric import BaseMetric
from deepeval.tracing.context import staged_next_span

try:
    from crewai import Agent, Crew, LLM

    crewai_installed = True
except ImportError:
    crewai_installed = False


def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


T = TypeVar("T")


def create_deepeval_class(
    base_class: Type[T],
    class_name: str,
    span_type: Optional[str],
    methods: tuple,
) -> Type[T]:
    """A CrewAI class whose ``methods`` stage its metrics for the span they
    open, of ``span_type``."""

    class DeepEvalClass(base_class):
        _metric_collection: Optional[str] = PrivateAttr(default=None)
        _metrics: Optional[List[BaseMetric]] = PrivateAttr(default=None)

        def __init__(self, *args, **kwargs):
            is_crewai_installed()
            metric_collection = kwargs.pop("metric_collection", None)
            metrics = kwargs.pop("metrics", None)
            super().__init__(*args, **kwargs)
            self._metric_collection = metric_collection
            self._metrics = metrics

    for method in methods:
        if hasattr(base_class, method):
            # Looked up on each call, to run as confident-trace patched it.
            resolve = functools.partial(getattr, base_class, method)
            setattr(DeepEvalClass, method, _staging(resolve, span_type))
    DeepEvalClass.__name__ = class_name
    DeepEvalClass.__qualname__ = class_name
    return DeepEvalClass


def create_deepeval_llm(base_factory: Callable) -> Callable:
    """Wrapper for factory functions/classes (LLM)."""

    def factory_wrapper(*args, **kwargs):
        is_crewai_installed()
        metric_collection = kwargs.pop("metric_collection", None)
        metrics = kwargs.pop("metrics", None)
        instance = base_factory(*args, **kwargs)
        try:
            instance._metric_collection = metric_collection
            instance._metrics = metrics
        except Exception:
            return instance
        # ``LLM`` builds a provider class, which makes the model calls.
        provider = type(instance)
        for method in ("call", "acall"):
            call = vars(provider).get(method)
            if call is not None and not getattr(
                call, "_deepeval_staging", False
            ):
                setattr(
                    provider, method, _staging(lambda call=call: call, "llm")
                )
        return instance

    return factory_wrapper


def _staging(
    resolve: Callable[[], Callable[..., Any]], span_type: Optional[str]
) -> Callable:
    """The method ``resolve`` returns, staging its object's metrics for the
    span it opens."""
    method = resolve()

    def fields(instance: Any) -> dict:
        return {
            "metrics": getattr(instance, "_metrics", None),
            "metric_collection": getattr(instance, "_metric_collection", None),
        }

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def staged(self, *args, **kwargs):
            with staged_next_span(span_type, fields(self)):
                return await resolve()(self, *args, **kwargs)

    else:

        @functools.wraps(method)
        def staged(self, *args, **kwargs):
            with staged_next_span(span_type, fields(self)):
                return resolve()(self, *args, **kwargs)

    staged._deepeval_staging = True
    return staged


if crewai_installed:
    DeepEvalCrew = create_deepeval_class(
        Crew, "DeepEvalCrew", None, ("kickoff", "akickoff")
    )
    DeepEvalAgent = create_deepeval_class(
        Agent,
        "DeepEvalAgent",
        "agent",
        (
            "execute_task",
            "aexecute_task",
            "_execute_and_build_output",
            "_execute_and_build_output_async",
        ),
    )
    DeepEvalLLM = create_deepeval_llm(LLM)
