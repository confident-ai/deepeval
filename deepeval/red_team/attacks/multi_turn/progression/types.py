from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from deepteam.attacks.multi_turn.progression.shift_detection import (
    MetricVerdict,
)
from deepteam.attacks.multi_turn.progression.stopping import StopReason
from deepteam.test_case.test_case import RTTurn


class Probe(BaseModel):
    id: int
    parent_id: Optional[int] = None
    input: str
    output: Optional[RTTurn] = None
    turn_level_attack: Optional[str] = None
    committed: bool = False
    depth: int = 0
    latency: Optional[float] = None
    simulation_cost: Optional[float] = None
    score: Optional[float] = None
    reason: Optional[str] = None


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
