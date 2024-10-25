from typing import List
from pydantic import BaseModel, Field


class ReasonScore(BaseModel):
    reasoning: str
    score: List[float]
