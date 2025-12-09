from __future__ import annotations
import uuid
from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
    TYPE_CHECKING,
    Union,
)
from enum import Enum
from pydantic import BaseModel, ConfigDict

from deepeval.prompt.prompt import Prompt

if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden

PromptConfigurationId = str
ModuleId = str
ScoreVector = List[float]  # scores per instance on D_pareto, aligned order
ScoreTable = Dict[PromptConfigurationId, ScoreVector]

# Type alias for model callback function
ModelCallback = Callable[[Prompt, Union["Golden", "ConversationalGolden"]], str]


@dataclass
class PromptConfiguration:
    id: PromptConfigurationId
    parent: Optional[PromptConfigurationId]
    prompts: Dict[ModuleId, Prompt]

    @staticmethod
    def new(
        prompts: Dict[ModuleId, Prompt],
        parent: Optional[PromptConfigurationId] = None,
    ) -> "PromptConfiguration":
        return PromptConfiguration(
            id=str(uuid.uuid4()), parent=parent, prompts=dict(prompts)
        )


class RunnerStatusType(str, Enum):
    """Status events emitted by optimization runners."""

    PROGRESS = "progress"
    TIE = "tie"
    ERROR = "error"


# Type alias for status callback function
RunnerStatusCallback = Callable[..., None]


class Objective(ABC):
    """Strategy for reducing scores per-metric to a single scalar value.

    Implementations receive a mapping from metric name to score
    (for example, {"AnswerRelevancyMetric": 0.82}) and return a
    single float used for comparisons inside the optimizer.
    """

    @abstractmethod
    def scalarize(self, scores_by_metric: Dict[str, float]) -> float:
        raise NotImplementedError


class MeanObjective(Objective):
    """Default scalarizer: unweighted arithmetic mean.

    - If `scores_by_metric` is non-empty, returns the arithmetic
      mean of all metric scores.
    - If `scores_by_metric` is empty, returns 0.0.
    """

    def scalarize(self, scores_by_metric: Dict[str, float]) -> float:
        if not scores_by_metric:
            return 0.0
        return sum(scores_by_metric.values()) / len(scores_by_metric)


class WeightedObjective(Objective):
    """
    Objective that scales each metric's score by a user-provided weight and sums them.

    - `weights_by_metric` keys should match the names of the metrics passed to the
      metric class names passed to the PromptOptimizer.
    - Metrics not present in `weights_by_metric` receive `default_weight`.
      This makes it easy to emphasize a subset of metrics while keeping
      everything else at a baseline weight of 1.0, e.g.:

          WeightedObjective({"AnswerRelevancyMetric": 2.0})

      which treats AnswerRelevancy as 2x as important as the other metrics.
    """

    def __init__(
        self,
        weights_by_metric: Optional[Dict[str, float]] = None,
        default_weight: float = 1.0,
    ):
        self.weights_by_metric: Dict[str, float] = dict(weights_by_metric or {})
        self.default_weight: float = float(default_weight)

    def scalarize(self, scores_by_metric: Dict[str, float]) -> float:
        return sum(
            self.weights_by_metric.get(name, self.default_weight) * score
            for name, score in scores_by_metric.items()
        )


class AcceptedIterationDict(TypedDict):
    parent: PromptConfigurationId
    child: PromptConfigurationId
    module: ModuleId
    before: float
    after: float


class AcceptedIteration(BaseModel):
    parent: str
    child: str
    module: str
    before: float
    after: float


class PromptConfigSnapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    parent: Optional[str]
    prompts: Dict[str, Prompt]


class OptimizationReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    optimization_id: str
    best_id: str
    accepted_iterations: List[AcceptedIteration]
    pareto_scores: Dict[str, List[float]]
    parents: Dict[str, Optional[str]]
    prompt_configurations: Dict[str, PromptConfigSnapshot]
