from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from deepeval.test_run.test_run import TestRun
from deepeval.utils import make_model_config


class ComparisonWinner(str, Enum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"
    TIE = "tie"


WinnerType = Literal["baseline", "candidate", "tie"]


class JudgeVerdict(BaseModel):
    """Raw structured decision emitted by the LLM judge for a single test case."""

    winner: WinnerType = Field(
        description="The winning run for this test case: 'baseline', 'candidate', or 'tie'."
    )
    reason: str = Field(
        description="Justification explaining why one output was superior or why they are equal."
    )

    model_config = make_model_config(arbitrary_types_allowed=True)


class TestCaseComparison(BaseModel):
    """Pairwise comparison result for an individual test case."""

    __test__ = False

    name: str | None = None
    input: str
    baseline_output: str | None = None
    candidate_output: str | None = None
    expected_output: str | None = None
    context: list[str] | None = None
    winner: Literal["baseline", "candidate", "tie"]
    reason: str

    model_config = make_model_config(arbitrary_types_allowed=True)


class TestRunComparisonResult(BaseModel):
    """Overall structured outcome of comparing two historical test runs."""

    __test__ = False

    winner: Literal["baseline", "candidate", "tie"]
    reason: str
    baseline_run: TestRun
    candidate_run: TestRun
    comparisons: list[TestCaseComparison] = Field(default_factory=list)
    baseline_wins: int = 0
    candidate_wins: int = 0
    ties: int = 0

    @property
    def total_comparisons(self) -> int:
        return len(self.comparisons)

    model_config = make_model_config(arbitrary_types_allowed=True)
