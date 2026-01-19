from typing import List, Optional, Type, TypeVar
from pydantic import PrivateAttr

from deepeval.metrics.base_metric import BaseMetric

try:
    from crewai import Crew, Agent, LLM

    is_crewai_installed = True
except ImportError:
    is_crewai_installed = False


def is_crewai_installed():
    if not is_crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


T = TypeVar("T")


def create_deepeval_class(
    base_class: Type[T], class_name: str, needs_new_override: bool = False
) -> Type[T]:
    """Factory function to create DeepEval-enabled CrewAI classes

    Args:
        base_class: The CrewAI class to extend
        class_name: Name for the new class
        needs_new_override: If True, override __new__ to filter kwargs before
                           they reach the parent's factory __new__ (needed for LLM)
    """

    class DeepEvalClass(base_class):
        _metric_collection: Optional[str] = PrivateAttr(default=None)
        _metrics: Optional[List[BaseMetric]] = PrivateAttr(default=None)

        def __init__(
            self,
            *args,
            metrics: Optional[List[BaseMetric]] = None,
            metric_collection: Optional[str] = None,
            **kwargs
        ):
            is_crewai_installed()
            # Retrieve stored values from __new__ if they were set there
            if metric_collection is None:
                metric_collection = getattr(
                    self.__class__, "_pending_metric_collection", None
                )
            if metrics is None:
                metrics = getattr(self.__class__, "_pending_metrics", None)

            super().__init__(*args, **kwargs)
            self._metric_collection = metric_collection
            self._metrics = metrics

            # Clean up class-level temp storage
            if hasattr(self.__class__, "_pending_metric_collection"):
                del self.__class__._pending_metric_collection
            if hasattr(self.__class__, "_pending_metrics"):
                del self.__class__._pending_metrics

    # Only add __new__ override for classes that need it (like LLM with factory pattern)
    if needs_new_override:

        def custom_new(
            cls,
            *args,
            metrics: Optional[List[BaseMetric]] = None,
            metric_collection: Optional[str] = None,
            **kwargs
        ):
            """
            Override __new__ to extract DeepEval-specific kwargs before they
            reach the parent class's __new__ method.

            CrewAI's LLM class uses __new__ as a factory that may pass kwargs
            to native provider classes that don't understand our custom params.
            """
            # Store DeepEval params on class for __init__ to retrieve
            cls._pending_metric_collection = metric_collection
            cls._pending_metrics = metrics

            # Call parent __new__ without DeepEval-specific kwargs
            return base_class.__new__(cls, *args, **kwargs)

        DeepEvalClass.__new__ = custom_new

    DeepEvalClass.__name__ = class_name
    DeepEvalClass.__qualname__ = class_name
    return DeepEvalClass


# Create the classes
# Crew and Agent use Pydantic models - no __new__ override needed
DeepEvalCrew = create_deepeval_class(Crew, "DeepEvalCrew")
DeepEvalAgent = create_deepeval_class(Agent, "DeepEvalAgent")
# LLM uses __new__ as a factory pattern - needs override to filter kwargs
DeepEvalLLM = create_deepeval_class(LLM, "DeepEvalLLM", needs_new_override=True)
