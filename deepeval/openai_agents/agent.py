from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, List

from deepeval.prompt import Prompt
from deepeval.metrics import BaseMetric
from deepeval.tracing.types import LlmSpan

try:
    from agents.agent import Agent as BaseAgent
    from deepeval.openai_agents.patch import (
        patch_default_agent_runner_get_model,
    )
except Exception as e:
    raise RuntimeError(
        "openai-agents is required for this integration. Please install it."
    ) from e

TContext = TypeVar("TContext")


@dataclass
class DeepEvalAgent(BaseAgent[TContext], Generic[TContext]):
    """
    A subclass of agents.Agent.
    """

    llm_metric_collection: str = None
    llm_metrics: List[BaseMetric] = None
    confident_prompt: Prompt = None
    agent_metrics: List[BaseMetric] = None
    agent_metric_collection: str = None

    def __post_init__(self):
        patch_default_agent_runner_get_model()
