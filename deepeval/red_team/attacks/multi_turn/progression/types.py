from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from deepeval.red_team.attacks.multi_turn.progression.shift_detection import (
    MetricVerdict,
)
from deepeval.red_team.attacks.multi_turn.progression.stopping import StopReason
from deepeval.red_team.test_case.test_case import Probe, RTTurn


class ProgressionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    turns: List[RTTurn]
    stop_reason: StopReason
    stop_detail: Optional[str] = None
    turns_spent: int = 0
    probes: List[Probe] = []
    shift_verdict: Optional[MetricVerdict] = None
    error: Optional[Exception] = None

    @property
    def last_response(self) -> Optional[str]:
        return self.turns[-1].content if self.turns else None
