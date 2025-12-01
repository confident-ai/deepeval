from __future__ import annotations
import uuid

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypedDict,
    Tuple,
    Union,
)
from enum import Enum
from pydantic import BaseModel as PydanticBaseModel, Field, AliasChoices

from deepeval.prompt.prompt import Prompt
from deepeval.models.base_model import DeepEvalBaseLLM


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden

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


class ScoringAdapter(Protocol):
    """
    Scoring adapter contract used by optimization runners.

    Runners call into this adapter to:
    - compute scores per-instance on some subset (score_on_pareto),
    - compute minibatch means for selection and acceptance,
    - generate feedback text used by the PromptRewriter.
    """

    # Sync
    def score_on_pareto(
        self,
        prompt_configuration: PromptConfiguration,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> ScoreVector:
        """Return per-instance scores on D_pareto."""
        ...

    def minibatch_score(
        self,
        prompt_configuration: PromptConfiguration,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        """Return average score μ on a minibatch from D_feedback."""
        ...

    def minibatch_feedback(
        self,
        prompt_configuration: PromptConfiguration,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        """Return μ_f text for the module (metric.reason + traces, etc.)."""
        ...

    def select_module(
        self, prompt_configuration: PromptConfiguration
    ) -> ModuleId:
        """Pick a module to mutate."""
        ...

    # Async
    async def a_score_on_pareto(
        self,
        prompt_configuration: PromptConfiguration,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> ScoreVector: ...
    async def a_minibatch_score(
        self,
        prompt_configuration: PromptConfiguration,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float: ...
    async def a_minibatch_feedback(
        self,
        prompt_configuration: PromptConfiguration,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str: ...
    async def a_select_module(
        self, prompt_configuration: PromptConfiguration
    ) -> ModuleId: ...


class PromptRewriterProtocol(Protocol):
    def rewrite(
        self,
        *,
        module_id: ModuleId,
        model: Optional[DeepEvalBaseLLM] = None,
        model_schema: Optional[PydanticBaseModel] = None,
        model_callback: Optional[
            Callable[
                ...,
                Union[
                    str,
                    Dict,
                    Tuple[Union[str, Dict], float],
                ],
            ]
        ] = None,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt: ...

    async def a_rewrite(
        self,
        *,
        module_id: ModuleId,
        model: Optional[DeepEvalBaseLLM] = None,
        model_schema: Optional[PydanticBaseModel] = None,
        model_callback: Optional[
            Callable[
                ...,
                Union[
                    str,
                    Dict,
                    Tuple[Union[str, Dict], float],
                ],
            ]
        ] = None,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt: ...


class RunnerStatusType(str, Enum):
    """Status events emitted by optimization runners."""

    PROGRESS = "progress"
    TIE = "tie"
    ERROR = "error"


class RunnerStatusCallbackProtocol(Protocol):
    def __call__(
        self,
        kind: RunnerStatusType,
        *,
        detail: str,
        step_index: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> None: ...


class RunnerProtocol(Protocol):
    """
    Contract for prompt optimization runners used by PromptOptimizer.

    Runners are responsible for executing the optimization algorithm
    and returning an optimized Prompt plus a report dict.
    """

    # status_callback is injected by PromptOptimizer
    # A runner may call this to report:
    # progress, ties, or errors during execution.
    status_callback: Optional[RunnerStatusCallbackProtocol]
    model_callback: Optional[
        Callable[
            ...,
            Union[
                str,
                Dict,
                Tuple[Union[str, Dict], float],
            ],
        ]
    ]

    scoring_adapter: Optional[ScoringAdapter]

    def execute(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, Dict]: ...

    async def a_execute(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, Dict]: ...


class Objective(Protocol):
    """Strategy for reducing scores per-metric to a single scalar value.

    Implementations receive a mapping from metric name to score
    (for example, {"AnswerRelevancyMetric": 0.82}) and return a
    single float used for comparisons inside the optimizer.
    """

    def scalarize(self, scores_by_metric: Dict[str, float]) -> float: ...


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


@dataclass
class MetricInfo:
    name: str
    rubric: Optional[str] = None


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
