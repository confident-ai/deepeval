from __future__ import annotations
from typing import Optional
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    PrivateAttr,
    confloat,
    conint,
)

from deepeval.optimization.policies.tiebreaker import (
    TieBreaker as TieBreakerPolicy,
)
from deepeval.optimization.types import PromptRewriter
from .mutation import NoOpRewriter


class GEPAAsyncConfig(BaseModel):
    """Async controls used by GEPA"""

    run_async: bool = True
    max_concurrent: PositiveInt = 20
    throttle_seconds: confloat(ge=0.0) = 0.0


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
        TieBreakerPolicy.PREFER_ROOT,
        description="How to break ties on aggregate",
    )
    async_config: GEPAAsyncConfig = Field(default_factory=GEPAAsyncConfig)
    display_options: GEPADisplayConfig = Field(
        default_factory=GEPADisplayConfig
    )
    _rewriter: Optional[PromptRewriter] = PrivateAttr(default=None)

    def get_rewriter(self) -> PromptRewriter:
        return self._rewriter or NoOpRewriter()


GEPAConfig.TieBreaker = TieBreakerPolicy
GEPAConfig.AsyncConfig = GEPAAsyncConfig
GEPAConfig.DisplayConfig = GEPADisplayConfig
