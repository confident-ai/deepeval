from __future__ import annotations
import uuid

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
    Union,
)
from enum import Enum
from pydantic import BaseModel, ConfigDict
from deepeval.prompt.prompt import Prompt
from deepeval.dataset.golden import Golden, ConversationalGolden

PromptConfigurationId = str
ModuleId = str
ScoreVector = List[float]
ScoreTable = Dict[PromptConfigurationId, ScoreVector]
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


RunnerStatusCallback = Callable[..., None]


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
