from typing import List, Literal
from pydantic import BaseModel


class StepVerdict(BaseModel):
    step_index: int
    verdict: Literal["PASS", "FAIL", "SKIP"]
    reason: str
    conclusion: str


class ProcessIntegrityResult(BaseModel):
    verdicts: List[StepVerdict]
    score: float
    reason: str
