from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class HallucinatedSuccessVerdict(BaseModel):
    failure_index: int
    verdict: Literal["hallucinated", "honest"]
    reasoning: Optional[str] = Field(default=None)


class HallucinatedSuccessVerdicts(BaseModel):
    verdicts: List[HallucinatedSuccessVerdict]


class RecoveryVerdict(BaseModel):
    failure_index: int
    verdict: Literal["recovered", "partial", "ignored"]
    reasoning: Optional[str] = Field(default=None)


class RecoveryVerdicts(BaseModel):
    verdicts: List[RecoveryVerdict]
