from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class AcceptedStepApi(BaseModel):
    parent: str
    child: str
    module: str
    before: float
    after: float


class OptimizationResultApi(BaseModel):
    optimization_id: str = Field(alias="optimizationId")
    best_id: str = Field(alias="bestId")
    accepted_steps: List[AcceptedStepApi] = Field(
        default_factory=list, alias="acceptedSteps"
    )
    pareto_scores: Dict[str, List[float]] = Field(alias="paretoScores")
    parents: Dict[str, Optional[str]]

    @classmethod
    def from_runtime(cls, result: Dict) -> "OptimizationResultApi":
        return cls(**result)
