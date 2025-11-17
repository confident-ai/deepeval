from __future__ import annotations
import time
from typing import Optional
from pydantic import (
    BaseModel,
    confloat,
    conint,
    Field,
    field_validator,
    PositiveInt,
    PrivateAttr,
)

from deepeval.optimization.policies.tiebreaker import (
    TieBreaker as TieBreakerPolicy,
)
from deepeval.optimization.types import PromptRewriterProtocol
from .mutation import PromptRewriter


class GEPADisplayConfig(BaseModel):
    """Display controls used by GEPA"""

    show_indicator: bool = True
    verbose: bool = False
    announce_ties: bool = Field(
        True, description="Print a one-line note when a tie is detected"
    )


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
    # Two candidates are considered tied if their aggregate scores are within tie_tolerance.
    tie_tolerance: confloat(ge=0.0) = Field(
        1e-9, description="Tie tolerance for aggregate scores"
    )
    tie_breaker: TieBreakerPolicy = Field(
        TieBreakerPolicy.RANDOM,
        description="How to break ties on aggregate",
    )

    display_options: GEPADisplayConfig = Field(
        default_factory=GEPADisplayConfig
    )
    _rewriter: Optional[PromptRewriterProtocol] = PrivateAttr(default=None)

    def set_rewriter(self, r: PromptRewriterProtocol) -> None:
        self._rewriter = r

    def get_rewriter(self) -> PromptRewriterProtocol:
        return self._rewriter or PromptRewriter()

    @field_validator("random_seed", mode="before")
    @classmethod
    def _coerce_random_seed(cls, seed):
        if seed is None:
            return time.time_ns()
        else:
            return seed


GEPAConfig.TieBreaker = TieBreakerPolicy
GEPAConfig.DisplayConfig = GEPADisplayConfig
