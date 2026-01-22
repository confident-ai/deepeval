import threading
from typing import Dict, List, Optional, Type, TypeVar
from pydantic import PrivateAttr

from deepeval.metrics.base_metric import BaseMetric

try:
    from crewai import Crew, Agent, LLM

    _crewai_installed = True
except ImportError:
    _crewai_installed = False


def is_crewai_installed():
    if not _crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


# Thread-local storage for passing DeepEval params from __new__ to __init__
# This avoids race conditions when multiple instances are created concurrently
_pending_params: threading.local = threading.local()


def _get_pending_params() -> Dict[str, Optional[object]]:
    """Get the thread-local pending params dict, creating if needed."""
    if not hasattr(_pending_params, "data"):
        _pending_params.data = {}
    return _pending_params.data


def _clear_pending_params() -> None:
    """Clear the thread-local pending params."""
    if hasattr(_pending_params, "data"):
        _pending_params.data = {}


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
            # Retrieve stored values from thread-local if set by __new__
            pending = _get_pending_params()
            if metric_collection is None:
                metric_collection = pending.get("metric_collection")
            if metrics is None:
                metrics = pending.get("metrics")

            super().__init__(*args, **kwargs)
            self._metric_collection = metric_collection
            self._metrics = metrics

            # Clean up thread-local storage
            _clear_pending_params()

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

            Uses thread-local storage to safely pass params to __init__.
            """
            # Store DeepEval params in thread-local for __init__ to retrieve
            pending = _get_pending_params()
            pending["metric_collection"] = metric_collection
            pending["metrics"] = metrics

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
