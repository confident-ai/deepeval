import logging

from typing import List, Optional, Type, TypeVar
from pydantic import PrivateAttr

from deepeval.config.settings import get_settings
from deepeval.metrics.base_metric import BaseMetric


logger = logging.getLogger(__name__)

try:
    from crewai import Crew, Agent, LLM

    _is_crewai_installed = True
except Exception as e:
    if logger.isEnabledFor(logging.DEBUG):
        show_trace = bool(get_settings().DEEPEVAL_LOG_STACK_TRACES)
        exc_info = (
            (type(e), e, getattr(e, "__traceback__", None))
            if show_trace
            else None
        )
        logger.debug("failed to import from crewai:", exc_info=exc_info)
    _is_crewai_installed = False


def is_crewai_installed():
    return _is_crewai_installed


def validate_crewai_installed():
    if not is_crewai_installed():
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


T = TypeVar("T")


def create_deepeval_class(base_class: Type[T], class_name: str) -> Type[T]:
    """Factory function to create DeepEval-enabled CrewAI classes"""

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
            validate_crewai_installed()
            super().__init__(*args, **kwargs)
            self._metric_collection = metric_collection
            self._metrics = metrics

    DeepEvalClass.__name__ = class_name
    DeepEvalClass.__qualname__ = class_name
    return DeepEvalClass


# Create the classes
DeepEvalCrew = create_deepeval_class(Crew, "DeepEvalCrew")
DeepEvalAgent = create_deepeval_class(Agent, "DeepEvalAgent")
DeepEvalLLM = create_deepeval_class(LLM, "DeepEvalLLM")
