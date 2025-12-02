from __future__ import annotations
import time
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    conint,
    confloat,
    field_validator,
)


class MIPROConfig(BaseModel):
    """
    Configuration for 0-shot MIPRO style prompt optimization.

    This is adapted to the DeepEval setting where we optimize a single Prompt
    (instruction) against a list of Goldens, using mini-batch evaluation and a
    simple surrogate over prompt candidates.

    Fields
    ------
    iterations:
        Total number of optimization trials. Each iteration selects
        a parent candidate, proposes a child via the PromptRewriter,
        evaluates it on a mini-batch, and updates the surrogate stats.

    minibatch_size:
        Fixed minibatch size drawn from the full set of goldens. When set,
        this overrides dynamic sizing based on `minibatch_ratio`,
        `minibatch_min_size`, and `minibatch_max_size`.

    minibatch_min_size:
        Hard lower bound on minibatch size when dynamic sizing is in effect.

    minibatch_max_size:
        Hard upper bound on minibatch size when dynamic sizing is in effect.

    minibatch_ratio:
        Target fraction of len(goldens) used to compute a dynamic minibatch
        size. The final size is bounded between `minibatch_min_size` and
        `minibatch_max_size`.

    random_seed:
        RNG seed for reproducibility. If set to None, a seed is derived from
        time.time_ns() by the validator.

    exploration_probability:
        Epsilon greedy exploration rate for candidate selection. With this
        probability the runner picks a random candidate; otherwise it picks
        the candidate with the highest mean minibatch score.

    full_eval_every:
        If set, every `full_eval_every` trials the runner fully evaluates the
        current best candidate (by mean minibatch score) on the full set of
        goldens, storing scores per-instance. If None, only a final full
        evaluation is done at the end.

    rewrite_instruction_max_chars:
        Maximum number of characters pulled into rewrite instructions
        (prompt text + feedback) when using PromptRewriter.

    min_delta:
        Minimum improvement on minibatch mean required for a child
        configuration to be accepted over its parent.
    """

    iterations: PositiveInt = Field(
        default=5,
        description="Total number of MIPRO trials or prompt proposals.",
    )
    minibatch_size: Optional[conint(ge=1)] = Field(
        default=None,
        description=(
            "Fixed minibatch size for goldens; when set, overrides dynamic sizing."
        ),
    )
    minibatch_min_size: conint(ge=1) = Field(
        default=4,
        description="Hard lower bound on minibatch size.",
    )
    minibatch_max_size: PositiveInt = Field(
        default=32,
        description="Hard upper bound on minibatch size.",
    )
    minibatch_ratio: confloat(gt=0.0, le=1.0) = Field(
        default=0.05,
        description=(
            "Target fraction of len(goldens) used to compute a dynamic minibatch "
            "size; bounded between minibatch_min_size and minibatch_max_size."
        ),
    )
    random_seed: conint(ge=0) = 0
    min_delta: confloat(ge=0.0) = Field(
        default=0.0,
        description=(
            "Minimum improvement in minibatch score required for a child "
            "prompt to be accepted over its parent."
        ),
    )

    exploration_probability: confloat(ge=0.0, le=1.0) = Field(
        default=0.2,
        description=(
            "Probability of sampling a random candidate instead of "
            "the best-by-mean minibatch score."
        ),
    )

    full_eval_every: Optional[PositiveInt] = Field(
        default=5,
        description=(
            "If set, the runner fully evaluates the current best candidate on the "
            "full goldens every N trials. If None, only a single full evaluation "
            "is performed at the end."
        ),
    )

    rewrite_instruction_max_chars: PositiveInt = Field(
        default=4096,
        description=(
            "Maximum number of characters from prompt, feedback, and related "
            "text included in rewrite instructions."
        ),
    )

    @field_validator("random_seed", mode="before")
    @classmethod
    def _coerce_random_seed(cls, seed):
        if seed is None:
            return time.time_ns()
        return seed
