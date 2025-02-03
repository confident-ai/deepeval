from typing import List, Optional
from pydantic import BaseModel


class OutOfCharacterResponseVerdict(BaseModel):
    index: int
    reason: str
    actual_output: Optional[str]


class OutOfCharacterResponseVerdicts(BaseModel):
    verdicts: List[OutOfCharacterResponseVerdict]


class Reason(BaseModel):
    reason: str
