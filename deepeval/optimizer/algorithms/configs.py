from pydantic import (
    BaseModel,
    Field,
    conint,
    PositiveInt,
    confloat,
    field_validator,
    Optional,
)
import time

from deepeval.optimizer.policies.tie_breaker import (
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


class MIPROV2Config(BaseModel):
    """
    Configuration for 0-shot MIPROV2 style prompt optimization.

    This is adapted to the DeepEval setting where we optimize a single Prompt
    (instruction) against a list of Goldens, using mini-batch evaluation and a
    simple surrogate over prompt candidates.

    Fields
    ------
    iterations:
        Total number of optimization trials. Each iteration selects
        a parent candidate, proposes a child via the Rewriter,
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
        (prompt text + feedback) when using Rewriter.

    min_delta:
        Minimum improvement on minibatch mean required for a child
        configuration to be accepted over its parent.
    """

    iterations: PositiveInt = Field(
        default=5,
        description="Total number of MIPROV2 trials or prompt proposals.",
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


class COPROConfig(MIPROV2Config):
    """
    Configuration for COPRO style cooperative prompt optimization.

    This extends MIPROV2Config with settings that control the cooperative
    sampling behavior.

    The core MIPROV2Config fields behave exactly the same as in MIPROv2.
    """

    population_size: conint(ge=1) = Field(
        default=4,
        description=(
            "Maximum number of prompt candidates maintained in the active pool. "
            "Once this limit is exceeded, lower scoring candidates are pruned."
        ),
    )

    proposals_per_step: conint(ge=1) = Field(
        default=4,
        description=(
            "Number of child prompts proposed cooperatively from the same "
            "parent in each optimization iteration."
        ),
    )


class SIMBAConfig(COPROConfig):
    """
    Configuration for SIMBA style cooperative prompt optimization.

    Extends `COPROConfig` with strategy specific controls:

      - How many minibatch examples are surfaced as demos and how long
        those snippets can be (`max_demos_per_proposal`,
        `demo_input_max_chars`).
    """

    max_demos_per_proposal: conint(ge=0) = Field(
        default=3,
        description=(
            "Maximum number of goldens from the current minibatch that are "
            "converted into concrete input/output demos when using the "
            "APPEND_DEMO strategy."
        ),
    )

    demo_input_max_chars: PositiveInt = Field(
        default=256,
        description=(
            "Maximum number of characters taken from the golden input and "
            "expected output when constructing demo snippets for APPEND_DEMO."
        ),
    )
