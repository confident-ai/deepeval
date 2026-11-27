from typing import List
from pydantic import BaseModel, Field


class ReasonScore(BaseModel):
    reason: str
    score: float = Field(ge=0, le=10)


class Steps(BaseModel):
    steps: List[str]
