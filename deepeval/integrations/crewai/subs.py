from typing import List, Optional, Type, TypeVar, Callable
from pydantic import PrivateAttr

from deepeval.metrics.base_metric import BaseMetric

try:
    from crewai import Crew, Agent, LLM

    crewai_installed = True
except ImportError:
    crewai_installed = False
    # Fallbacks so the factory calls at the bottom of this module do not raise a
    # NameError at import time when crewai is absent. `Crew`/`Agent` are used as
    # base classes (must be classes); `LLM` is only invoked lazily inside the
    # factory wrapper, after the is_crewai_installed() guard has run.
    Crew = Agent = object
    LLM = None


def is_crewai_installed():
    # NOTE: this must check the `crewai_installed` flag, not a name that shadows
    # this function. Previously the flag was also named `is_crewai_installed`,
    # so `not is_crewai_installed` tested this function object (always truthy)
    # and the ImportError was never raised.
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


T = TypeVar("T")


def create_deepeval_class(base_class: Type[T], class_name: str) -> Type[T]:
    """Factory function to create DeepEval-enabled CrewAI classes"""

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
            pass
        return instance

    return factory_wrapper


DeepEvalCrew = create_deepeval_class(Crew, "DeepEvalCrew")
DeepEvalAgent = create_deepeval_class(Agent, "DeepEvalAgent")
DeepEvalLLM = create_deepeval_llm(LLM)
