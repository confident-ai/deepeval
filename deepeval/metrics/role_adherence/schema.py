from typing import List, Optional
from pydantic import BaseModel, Field


class OutOfCharacterResponseVerdict(BaseModel):
    index: int
    reason: str
    actual_output: Optional[str] = Field(default=None)


class OutOfCharacterResponseVerdicts(BaseModel):
    verdicts: List[OutOfCharacterResponseVerdict]


class Reason(BaseModel):
    reason: str
