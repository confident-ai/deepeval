from pydantic import BaseModel
from typing import List, Optional, Literal


class LoopTrigger(BaseModel):
    type: Literal["tool_repeat", "call_cycle", "reasoning_stagnation"]
    tool: Optional[str] = None
    steps: List[int]
    args_fingerprint: Optional[str] = None
    description: str


class LoopDetectionVerdict(BaseModel):
    score: float
    reason: str
    loop_triggers: List[LoopTrigger]
