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

    This controls:
      - The iteration budget and acceptance threshold (iterations, min_delta).
      - How D_train is split into a Pareto validation subset (D_pareto)
        versus a feedback subset (D_feedback) (pareto_size).
      - How minibatches are drawn from D_feedback, either with a fixed size
        or dynamically from a ratio and min/max bounds (minibatch_* fields).
      - How ties on aggregate scores are treated (tie_tolerance, tie_breaker).
      - Randomness and rewrite instruction length (random_seed,
        rewrite_instruction_max_chars).

    See individual field descriptions for precise behavior.
    """

    iterations: PositiveInt = Field(
        default=5,
        description="Total number of GEPA loop iterations (mutation attempts). "
        "This acts as the optimization budget B in the GEPA paper.",
    )
    minibatch_size: Optional[conint(ge=1)] = Field(
        default=None,
        description="Fixed minibatch size drawn from D_feedback. When set, this "
        "overrides dynamic sizing based on `minibatch_ratio`, "
        "`minibatch_min_size`, and `minibatch_max_size`.",
    )
    minibatch_min_size: conint(ge=1) = Field(
        default=4,
        description="Hard lower bound on the minibatch size used for D_feedback "
        "when dynamic sizing is in effect.",
    )
    minibatch_max_size: PositiveInt = Field(
        default=32,
        description="Hard upper bound on the minibatch size used for D_feedback "
        "when dynamic sizing is in effect.",
    )
    minibatch_ratio: confloat(gt=0.0, le=1.0) = Field(
        default=0.05,
        description=(
            "Target fraction of |D_feedback| used to compute a dynamic "
            "minibatch size when `minibatch_size` is None. The effective "
            "size is round(len(D_feedback) * minibatch_ratio) bounded "
            "between `minibatch_min_size` and `minibatch_max_size` and not "
            "exceeding len(D_feedback). D_feedback is the subset of the "
            "provided goldens that is not allocated to D_pareto by "
            "`split_goldens(...)`."
        ),
    )
    pareto_size: conint(ge=1) = Field(
        default=3,
        description="Size of the Pareto validation subset D_pareto. The splitter "
        "will bind this between [0, len(goldens)], and the runner requires "
        "at least 2 total goldens to run GEPA.",
    )
    random_seed: conint(ge=0) = Field(
        default=0,
        description="Non-negative RNG seed for reproducibility. "
        "If you explicitly pass None, it is replaced with a seed "
        "derived from time.time_ns() via the field validator.",
    )
    min_delta: confloat(ge=0.0) = Field(
        default=0.0,
        description="Minimum improvement required for a child configuration to be "
        "accepted, e.g. σ_child >= σ_parent + min_delta. A small jitter "
        "is applied internally to avoid floating-point edge cases.",
    )
    # Two candidates are considered tied if their aggregate scores are within tie_tolerance.
    tie_tolerance: confloat(ge=0.0) = Field(
        1e-9,
        description="Two candidates are considered tied on aggregate score if "
        "their values differ by at most this tolerance.",
    )
    tie_breaker: TieBreakerPolicy = Field(
        TieBreakerPolicy.PREFER_CHILD,
        description="Policy used to break ties when multiple prompt configurations "
        "share the best aggregate score. See `GEPAConfig.TieBreaker` "
        "for the available options. ",
    )
    rewrite_instruction_max_chars: PositiveInt = Field(
        default=4096,
        description=(
            "Maximum number of characters from prompt, feedback, and related text "
            "included in rewrite instructions."
        ),
    )

    @field_validator("random_seed", mode="before")
    @classmethod
    def _coerce_random_seed(cls, seed):
        if seed is None:
            return time.time_ns()
        else:
            return seed


GEPAConfig.TieBreaker = TieBreakerPolicy
