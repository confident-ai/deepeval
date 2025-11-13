import logging
import warnings

from importlib import import_module
from typing import List, Optional, Type, TypeVar, Tuple
from pydantic import PrivateAttr

from deepeval.runtime import py_version_str, pkg_version
from deepeval.config.settings import get_settings
from deepeval.metrics.base_metric import BaseMetric


logger = logging.getLogger(__name__)


class CrewAIIntegrationWarning(UserWarning):
    pass


EVENT_SYMBOLS = (
    "BaseEventListener",
    "CrewKickoffStartedEvent",
    "CrewKickoffCompletedEvent",
    "LLMCallStartedEvent",
    "LLMCallCompletedEvent",
    "AgentExecutionStartedEvent",
    "AgentExecutionCompletedEvent",
    "ToolUsageStartedEvent",
    "ToolUsageFinishedEvent",
    "KnowledgeRetrievalStartedEvent",
    "KnowledgeRetrievalCompletedEvent",
)


warnings.filterwarnings("once", category=CrewAIIntegrationWarning)


def _log_debug_exception(msg: str, exc: BaseException):
    s = get_settings()
    show = logger.isEnabledFor(logging.DEBUG) and bool(
        s.DEEPEVAL_LOG_STACK_TRACES
    )
    logger.debug(
        msg, exc_info=((type(exc), exc, exc.__traceback__) if show else None)
    )


def _warn(
    why: str, exc: Optional[BaseException] = None, *, context: str
) -> None:
    """
    Single, user-friendly warning:
      - no Poetry-specific instructions
      - no hard-coded Python version ranges
      - shows how to enable debug + stack traces
      - prints detected versions (diagnostic only)
    """
    s = get_settings()
    msg = (
        "DeepEval CrewAI integration is unavailable: " + why + "\n\n"
        "How to enable:\n"
        "  • Install CrewAI: `pip install crewai`\n"
        "  • For more details: set `LOG_LEVEL=debug` and `DEEPEVAL_LOG_STACK_TRACES=1`\n\n"
        f"Detected: Python={py_version_str()}, crewai={pkg_version('crewai') or 'not installed'}, "
        f"LOG_LEVEL={logging.getLevelName(logger.getEffectiveLevel()).lower()}, "
        f"DEEPEVAL_LOG_STACK_TRACES={bool(s.DEEPEVAL_LOG_STACK_TRACES)}, context={context}"
    )
    warnings.warn(msg, category=CrewAIIntegrationWarning, stacklevel=3)

    if (
        exc
        and logger.isEnabledFor(logging.DEBUG)
        and s.DEEPEVAL_LOG_STACK_TRACES
    ):
        logger.debug(f"CrewAI import failure in {context}", exc_info=True)


def is_crewai_installed() -> bool:
    return pkg_version("crewai") is not None


def validate_crewai_installed():
    if not is_crewai_installed():
        raise ImportError(
            "CrewAI is not installed and is required to use deepeval crewai integration."
        )


def try_import_events():
    """Return event classes in a fixed order matching EVENT_SYMBOLS, or Nones."""
    try:
        events_mod = import_module("crewai.events")
        values = tuple(
            getattr(events_mod, name, None) for name in EVENT_SYMBOLS
        )
        if any(v is None for v in values):
            _warn(
                "`crewai` is installed but expected event symbols were not found.",
                None,
                context="events",
            )
        return values
    except ModuleNotFoundError as e:
        _warn(
            "`crewai` is not installed (optional dependency).",
            e,
            context="events",
        )
    except ImportError as e:
        _warn("`crewai.events` import failed.", e, context="events")
    except Exception as e:
        _warn(
            "Unexpected error while importing `crewai.events`.",
            e,
            context="events",
        )
    return (None,) * len(EVENT_SYMBOLS)


def try_import_tool_decorator():
    try:
        from crewai.tools import tool as crewai_tool

        return crewai_tool
    except ModuleNotFoundError as e:
        _warn(
            "`crewai` is not installed (optional dependency).",
            e,
            context="tools",
        )
    except ImportError as e:
        _warn(
            "`crewai.tools.tool` not found in installed version.",
            e,
            context="tools",
        )
    except Exception as e:
        _warn(
            "Unexpected error while importing `crewai.tools`.",
            e,
            context="tools",
        )
    return None


def try_import_core_classes() -> Tuple[object, object, object]:
    try:
        from crewai.crew import Crew
        from crewai.agent import Agent
        from crewai.llm import LLM

        return Crew, Agent, LLM
    except ModuleNotFoundError as e:
        _warn(
            "`crewai` is not installed (optional dependency).",
            e,
            context="core",
        )
    except ImportError as e:
        _warn(
            "`crewai` API changed; core classes not found.", e, context="core"
        )
    except Exception as e:
        _warn(
            "Unexpected error while importing core `crewai` classes.",
            e,
            context="core",
        )
    return (None, None, None)


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
            **kwargs,
        ):
            validate_crewai_installed()
            super().__init__(*args, **kwargs)
            self._metric_collection = metric_collection
            self._metrics = metrics

    DeepEvalClass.__name__ = class_name
    DeepEvalClass.__qualname__ = class_name
    return DeepEvalClass


# Create the classes
DeepEvalCrew = DeepEvalAgent = DeepEvalLLM = None
Crew, Agent, LLM = try_import_core_classes()
if all([Crew, Agent, LLM]):
    DeepEvalCrew = create_deepeval_class(Crew, "DeepEvalCrew")
    DeepEvalAgent = create_deepeval_class(Agent, "DeepEvalAgent")
    DeepEvalLLM = create_deepeval_class(LLM, "DeepEvalLLM")
