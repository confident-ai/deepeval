from __future__ import annotations
import uuid
from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
)
from enum import Enum
from pydantic import (
    BaseModel as PydanticBaseModel,
    Field,
    AliasChoices,
)

from deepeval.prompt.prompt import Prompt

PromptConfigurationId = str
ModuleId = str
ScoreVector = List[float]  # scores per instance on D_pareto, aligned order
ScoreTable = Dict[PromptConfigurationId, ScoreVector]


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


class AcceptedIteration(PydanticBaseModel):
    parent: str
    child: str
    module: str
    before: float
    after: float


class PromptMessageSnapshot(PydanticBaseModel):
    role: str
    content: str


class PromptModuleSnapshot(PydanticBaseModel):
    type: Literal["TEXT", "LIST"]
    # Only used when type == "TEXT"
    text_template: Optional[str] = None
    # Only used when type == "LIST"
    messages: Optional[List[PromptMessageSnapshot]] = None


class PromptConfigSnapshot(PydanticBaseModel):
    parent: Optional[str]
    prompts: Dict[str, PromptModuleSnapshot]


@dataclass
class OptimizationResult:
    optimization_id: str
    best_id: PromptConfigurationId
    accepted_iterations: List[Dict]
    pareto_scores: Dict[PromptConfigurationId, List[float]]
    parents: Dict[PromptConfigurationId, Optional[PromptConfigurationId]]
    prompt_configurations: Dict[PromptConfigurationId, Dict[str, Any]]

    def as_dict(self) -> Dict:
        return dict(
            optimization_id=self.optimization_id,
            best_id=self.best_id,
            accepted_iterations=self.accepted_iterations,
            pareto_scores=self.pareto_scores,
            parents=self.parents,
            prompt_configurations=self.prompt_configurations,
        )


class OptimizationReport(PydanticBaseModel):
    optimization_id: str = Field(
        alias="optimizationId",
        validation_alias=AliasChoices("optimizationId", "optimization_id"),
    )
    best_id: str = Field(
        alias="bestId",
        validation_alias=AliasChoices("bestId", "best_id"),
    )
    accepted_iterations: list[AcceptedIteration] = Field(
        default_factory=list,
        alias="acceptedIterations",
        validation_alias=AliasChoices(
            "acceptedIterations", "accepted_iterations"
        ),
    )
    pareto_scores: dict[str, list[float]] = Field(
        alias="paretoScores",
        validation_alias=AliasChoices("paretoScores", "pareto_scores"),
    )
    parents: dict[str, str | None]
    prompt_configurations: dict[str, PromptConfigSnapshot] = Field(
        alias="promptConfigurations",
        validation_alias=AliasChoices(
            "promptConfigurations", "prompt_configurations"
        ),
    )

    @classmethod
    def from_runtime(cls, result: dict) -> "OptimizationReport":
        # accepts the dict from OptimizationResult.as_dict()
        return cls(**result)
