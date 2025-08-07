from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ArgumentCorrectnessVerdict(BaseModel):
    verdict: Literal["yes", "no", "idk"]
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ArgumentCorrectnessVerdict]


class ArgumentCorrectnessScoreReason(BaseModel):
    reason: str
