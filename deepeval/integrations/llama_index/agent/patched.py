from typing import TypeVar, cast, Optional, List
from pydantic import Field
from deepeval.metrics import BaseMetric
from deepeval.telemetry import capture_tracing_integration

try:
    from llama_index.core.agent.workflow import (
        FunctionAgent,
        ReActAgent,
        CodeActAgent,
    )

    is_llama_index_installed = True
except:
    is_llama_index_installed = False


def is_llama_index_agent_installed():
    if not is_llama_index_installed:
        raise ImportError(
            "llama-index is neccesary for this functionality. Please install it with `pip install llama-index` or with package manager of choice."
        )


T = TypeVar("T", bound=type)


def with_metrics(cls: T) -> T:
    class SubClassWithMetric(cls):  # type: ignore
        metric_collection: Optional[str] = Field(default=None)
        metrics: Optional[List[BaseMetric]] = Field(default_factory=list)

    SubClassWithMetric.__name__ = cls.__name__
    SubClassWithMetric.__qualname__ = cls.__qualname__
    return cast(T, SubClassWithMetric)


@with_metrics
class FunctionAgent(FunctionAgent):
    def __init__(self, *args, metric_collection=None, metrics=None, **kwargs):
        with capture_tracing_integration(
            "llama_index.agent.patched.FunctionAgent"
        ):
            super().__init__(*args, **kwargs)
            self.metric_collection = metric_collection
            self.metrics = metrics


@with_metrics
class ReActAgent(ReActAgent):
    def __init__(self, *args, metric_collection=None, metrics=None, **kwargs):
        with capture_tracing_integration(
            "llama_index.agent.patched.ReActAgent"
        ):
            super().__init__(*args, **kwargs)
            self.metric_collection = metric_collection
            self.metrics = metrics


@with_metrics
class CodeActAgent(CodeActAgent):
    def __init__(self, *args, metric_collection=None, metrics=None, **kwargs):
        with capture_tracing_integration(
            "llama_index.agent.patched.CodeActAgent"
        ):
            super().__init__(*args, **kwargs)
            self.metric_collection = metric_collection
            self.metrics = metrics
