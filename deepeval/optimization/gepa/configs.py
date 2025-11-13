from __future__ import annotations
from typing import Optional
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    confloat,
    conint,
)

from deepeval.optimization.types import PromptRewriter
from .mutation import NoOpRewriter


class GEPAConfig(BaseModel):
    """
    Core GEPA hyperparameters.
    - budget: total mutation attempts (Alg. 1 loop steps)
    - minibatch_size: b is size of minibatch from D_feedback
    - pareto_size: n_pareto is size of D_pareto
    - random_seed: RNG seed for reproducibility
    - min_delta: optional acceptance tolerance, such as σ′ >= σ + min_delta
    """

    budget: PositiveInt = Field(..., description="Total mutation attempts")
    minibatch_size: PositiveInt = Field(
        ..., description="Minibatch size for D_feedback"
    )
    pareto_size: conint(ge=1) = Field(
        ..., description="Size of D_pareto (must be >= 1)"
    )
    random_seed: int = 0
    min_delta: confloat(ge=0.0) = 0.0
    rewriter: Optional[PromptRewriter] = None

    def get_rewriter(self) -> PromptRewriter:
        return self.rewriter or NoOpRewriter()
