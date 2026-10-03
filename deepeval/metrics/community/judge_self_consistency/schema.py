from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class Replicate(BaseModel):
    """One repeat of the wrapped judge on the same test case."""

    score: Optional[float] = None
    success: Optional[bool] = None
    reason: Optional[str] = None
    error: Optional[str] = None


class JudgeSelfConsistencyResult(BaseModel):
    """The full reliability reading across all replicates."""

    stability: float = Field(
        description="1 - normalized variance of the replicate scores, in [0, 1]."
    )
    decision_flip_rate: Optional[float] = Field(
        default=None,
        description=(
            "Fraction of replicate pairs that landed on opposite sides of the "
            "wrapped judge's threshold. None when the judge has no threshold."
        ),
    )
    mean_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    score_interval: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Percentile bootstrap confidence interval for the mean score.",
    )
    replicates: List[Replicate] = Field(default_factory=list)
    errored_replicates: int = 0
