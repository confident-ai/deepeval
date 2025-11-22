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
)

from deepeval.optimization.policies.tie_breaker import (
    TieBreaker as TieBreakerPolicy,
)


class GEPAConfig(BaseModel):
    """
    Core configuration for the GEPA optimization loop.

    Fields:
      - iterations:
          Total number of GEPA loop iterations (mutation attempts).

      - minibatch_size:
          Fixed minibatch size drawn from D_feedback. When set, this
          overrides dynamic sizing based on `minibatch_ratio`,
          `minibatch_min_size`, and `minibatch_max_size`.

      - minibatch_min_size:
          Hard lower bound on the minibatch size used for D_feedback
          when dynamic sizing is in effect.

      - minibatch_max_size:
          Hard upper bound on the minibatch size used for D_feedback
          when dynamic sizing is in effect.

      - minibatch_ratio:
          Target fraction of len(goldens) used to compute a dynamic
          minibatch size for D_feedback. The final size is bound
          between `minibatch_min_size` and `minibatch_max_size`.

      - pareto_size:
          Size of the Pareto validation subset D_pareto. The splitter
          will bind this between [0, len(goldens)], and the runner requires
          at least 2 total goldens to run GEPA.

      - random_seed:
          RNG seed for reproducibility. If set to None, a seed is
          derived from time.time_ns() via the field validator.

      - min_delta:
          Minimum improvement required for a child configuration to be
          accepted, e.g. σ_child >= σ_parent + min_delta. A small jitter
          is applied internally to avoid floating-point edge cases.

      - tie_tolerance:
          Two candidates are considered tied on aggregate score if
          their values differ by at most this tolerance.

      - tie_breaker:
          Policy used to break ties when multiple prompt configurations
          share the best aggregate score. See `GEPAConfig.TieBreaker`
          for the available options.
    """

    iterations: PositiveInt = Field(
        default=5, description="Total mutation attempts"
    )
    minibatch_size: Optional[conint(ge=1)] = Field(
        default=None,
        description="Fixed minibatch size for D_feedback; when set, overrides dynamic sizing.",
    )
    minibatch_min_size: conint(ge=1) = Field(
        default=4,
        description="Hard lower bound on minibatch size for D_feedback.",
    )
    minibatch_max_size: PositiveInt = Field(
        default=32,
        description="Hard upper bound on minibatch size for D_feedback.",
    )
    minibatch_ratio: confloat(gt=0.0, le=1.0) = Field(
        default=0.05,
        description=(
            "Target fraction of len(goldens) used to compute a dynamic "
            "minibatch size for D_feedback."
        ),
    )
    pareto_size: conint(ge=1) = Field(
        default=3, description="Size of D_pareto (must be >= 1)"
    )
    random_seed: int = 0
    min_delta: confloat(ge=0.0) = 0.0
    # Two candidates are considered tied if their aggregate scores are within tie_tolerance.
    tie_tolerance: confloat(ge=0.0) = Field(
        1e-9, description="Tie tolerance for aggregate scores"
    )
    tie_breaker: TieBreakerPolicy = Field(
        TieBreakerPolicy.PREFER_CHILD,
        description="How to break ties on aggregate",
    )

    @field_validator("random_seed", mode="before")
    @classmethod
    def _coerce_random_seed(cls, seed):
        if seed is None:
            return time.time_ns()
        else:
            return seed


GEPAConfig.TieBreaker = TieBreakerPolicy
