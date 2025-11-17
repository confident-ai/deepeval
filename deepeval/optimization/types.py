from __future__ import annotations
import uuid

from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypedDict,
    Union,
)
from pydantic import BaseModel, Field, AliasChoices

from deepeval.prompt.prompt import Prompt
from deepeval.models.base_model import DeepEvalBaseLLM


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


# move inline make union
CandidateId = str
ModuleId = str
ScoreVector = List[float]  # scores per instance on D_pareto, aligned order
ScoreTable = Dict[CandidateId, ScoreVector]


@dataclass
class Candidate:
    id: CandidateId
    parent: Optional[CandidateId]
    prompts: Dict[ModuleId, Prompt]

    @staticmethod
    def new(
        prompts: Dict[ModuleId, Prompt],
        parent: Optional[CandidateId] = None,
    ) -> "Candidate":
        return Candidate(
            id=str(uuid.uuid4()), parent=parent, prompts=dict(prompts)
        )


class ScoringAdapter(Protocol):
    """Scoring Adapter contract used by GEPARunner (sync/async twins)."""

    # Sync
    def score_on_pareto(
        self,
        candidate: Candidate,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> ScoreVector:
        """Return per-instance scores on D_pareto."""
        ...

    def minibatch_score(
        self,
        candidate: Candidate,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        """Return average score μ on a minibatch from D_feedback."""
        ...

    def minibatch_feedback(
        self,
        candidate: Candidate,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        """Return μ_f text for the module (metric.reason + traces, etc.)."""
        ...

    def select_module(self, candidate: Candidate) -> ModuleId:
        """Pick a module to mutate (random/weighted/round-robin)."""
        ...

    # Async
    async def a_score_on_pareto(
        self,
        candidate: Candidate,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> ScoreVector: ...
    async def a_minibatch_score(
        self,
        candidate: Candidate,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float: ...
    async def a_minibatch_feedback(
        self,
        candidate: Candidate,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str: ...
    async def a_select_module(self, candidate: Candidate) -> ModuleId: ...


class PromptRewriterProtocol(Protocol):
    def rewrite(
        self,
        *,
        model: Optional[DeepEvalBaseLLM],
        module_id: ModuleId,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt: ...

    async def a_rewrite(
        self,
        *,
        model: Optional[DeepEvalBaseLLM],
        module_id: ModuleId,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt: ...


class Objective(Protocol):
    """Scalarizes per-metric scores into a single μ per instance."""

    def scalarize(self, scores_by_metric: Dict[str, float]) -> float: ...


class MeanObjective(Objective):
    def scalarize(self, scores_by_metric: Dict[str, float]) -> float:
        if not scores_by_metric:
            return 0.0
        return sum(scores_by_metric.values()) / len(scores_by_metric)


class WeightedObjective(Objective):
    def __init__(self, weights_by_metric: Dict[str, float]):
        self.weights_by_metric = dict(weights_by_metric)

    def scalarize(self, scores_by_metric: Dict[str, float]) -> float:
        return sum(
            self.weights_by_metric.get(name, 0.0)
            * scores_by_metric.get(name, 0.0)
            for name in scores_by_metric.keys()
        )


@dataclass
class MetricInfo:
    name: str
    rubric: Optional[str] = None


class AcceptedStepDict(TypedDict):
    parent: CandidateId
    child: CandidateId
    module: ModuleId
    before: float
    after: float


class AcceptedStep(BaseModel):
    parent: str
    child: str
    module: str
    before: float
    after: float


@dataclass
class OptimizationResult:
    optimization_id: str
    best_id: CandidateId
    accepted_steps: List[Dict]
    pareto_scores: Dict[CandidateId, List[float]]
    parents: Dict[CandidateId, Optional[CandidateId]]

    def as_dict(self) -> Dict:
        return dict(
            optimization_id=self.optimization_id,
            best_id=self.best_id,
            accepted_steps=self.accepted_steps,
            pareto_scores=self.pareto_scores,
            parents=self.parents,
        )


class OptimizationReport(BaseModel):
    optimization_id: str = Field(
        alias="optimizationId",
        validation_alias=AliasChoices("optimizationId", "optimization_id"),
    )
    best_id: str = Field(
        alias="bestId",
        validation_alias=AliasChoices("bestId", "best_id"),
    )
    accepted_steps: list[AcceptedStep] = Field(
        default_factory=list,
        alias="acceptedSteps",
        validation_alias=AliasChoices("acceptedSteps", "accepted_steps"),
    )
    pareto_scores: dict[str, list[float]] = Field(
        alias="paretoScores",
        validation_alias=AliasChoices("paretoScores", "pareto_scores"),
    )
    parents: dict[str, str | None]

    @classmethod
    def from_runtime(cls, result: dict) -> "OptimizationReport":
        # accepts the dict from OptimizationResult.as_dict()
        return cls(**result)
