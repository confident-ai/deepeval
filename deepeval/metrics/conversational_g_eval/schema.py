from typing import List
from pydantic import BaseModel


class ReasonScore(BaseModel):
    reason: str
    score: float


class Steps(BaseModel):
    steps: List[str]
