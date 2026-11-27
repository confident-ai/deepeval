from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from deepeval.red_team.attacks.multi_turn.progression import ProgressionResult
from deepeval.red_team.orchestration.schema import (
    CriteriaVerdict,
    SequentialVerdict,
)
from deepeval.red_team.test_case.test_case import RTTurn


class Escalation(BaseModel):
    turn: int
    guidance: str
    strategy: Optional[str] = None


class SubAgentReport(BaseModel):
    criteria: str
    scenario: str
    target_analysis: str
    instructions: str
    strategy: Optional[str] = None
    progression: ProgressionResult
    verdicts: List[CriteriaVerdict] = []
    escalations: List[Escalation] = []


class MetCriteria(BaseModel):
    criteria: str
    how_it_was_met: str
    round: int


class RoundResult(BaseModel):
    number: int
    scenario: str
    reports: List[SubAgentReport] = []
    winner: Optional[SubAgentReport] = None


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # None when the run failed during recon; see `error`.
    recon: Optional[ProgressionResult] = None
    analysis: str = ""
    criteria: List[str] = []
    rounds: List[RoundResult] = []
    met_criteria: List[MetCriteria] = []
    conversation: List[RTTurn] = []
    error: Optional[Exception] = None

    @property
    def score(self) -> float:
        if not self.criteria:
            return 0.0
        return len(self.met_criteria) / len(self.criteria)


class SequentialResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: str
    scenario: str
    criteria: List[str] = []
    turns: List[RTTurn] = []
    verdicts: List[SequentialVerdict] = []
    met_criteria: List[str] = []
    error: Optional[Exception] = None

    @property
    def score(self) -> float:
        if not self.criteria:
            return 0.0
        return len(self.met_criteria) / len(self.criteria)
