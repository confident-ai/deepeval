from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CalibrationBin(BaseModel):
    """One equal-width confidence bin. Mirrors ``BinStat`` from the math core.

    Bins are ``[lower, upper)``; the last bin is closed so confidence 1.0 is
    included. For an empty bin (``count == 0``) the math core reports
    ``avg_confidence == accuracy == 0.0`` as placeholders; ``count`` is what
    says whether a bin holds data, so consumers must check it first.
    """

    model_config = ConfigDict(frozen=True)

    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    avg_confidence: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_bin(self) -> "CalibrationBin":
        if self.upper <= self.lower:
            raise ValueError("upper must be greater than lower")
        if self.count == 0 and (
            self.avg_confidence != 0.0 or self.accuracy != 0.0
        ):
            raise ValueError(
                "an empty bin must have avg_confidence == accuracy == 0.0"
            )
        return self


class CalibrationResult(BaseModel):
    """Aggregate calibration over the rows seen so far in one evaluation run.

    Structure only: all numbers are computed elsewhere. The result is a
    snapshot (frozen); the metric recomputes a new one as rows arrive.
    """

    model_config = ConfigDict(frozen=True)

    ece: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    n_bins: int = Field(ge=1)
    n_samples: int = Field(ge=1)
    # Required on purpose: which event was scored must never be implicit.
    outcome_definition: str = Field(min_length=1)
    bins: List[CalibrationBin] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    reasoning: str = ""

    @model_validator(mode="after")
    def _check_bins(self) -> "CalibrationResult":
        # bins may be empty (caller opted out of the breakdown); if present
        # they must cover every bin and account for every sample.
        if self.bins:
            if len(self.bins) != self.n_bins:
                raise ValueError("bins must contain exactly n_bins entries")
            if sum(b.count for b in self.bins) != self.n_samples:
                raise ValueError("bin counts must sum to n_samples")
        return self
