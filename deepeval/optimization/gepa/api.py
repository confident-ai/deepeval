from pydantic import BaseModel, Field, AliasChoices
from typing import Dict, List, Optional


class AcceptedStepApi(BaseModel):
    parent: str
    child: str
    module: str
    before: float
    after: float


class OptimizationResultApi(BaseModel):
    optimization_id: str = Field(
        alias="optimizationId",
        validation_alias=AliasChoices("optimizationId", "optimization_id"),
    )
    best_id: str = Field(
        alias="bestId",
        validation_alias=AliasChoices("bestId", "best_id"),
    )
    accepted_steps: List[AcceptedStepApi] = Field(
        default_factory=list,
        alias="acceptedSteps",
        validation_alias=AliasChoices("acceptedSteps", "accepted_steps"),
    )
    pareto_scores: Dict[str, List[float]] = Field(
        alias="paretoScores",
        validation_alias=AliasChoices("paretoScores", "pareto_scores"),
    )
    parents: Dict[str, Optional[str]]

    @classmethod
    def from_runtime(cls, result: Dict) -> "OptimizationResultApi":
        return cls(**result)


# TODO: probably just rename the above
class OptimizationReport(OptimizationResultApi):
    """Optimization report with the same schema as OptimizationResultApi."""

    pass
