from typing import List
from pydantic import BaseModel


class ReasonScore(BaseModel):
    reasoning: str
    score: float
