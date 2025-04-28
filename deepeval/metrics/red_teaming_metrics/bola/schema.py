from typing import List

from pydantic import BaseModel


class Entities(BaseModel):
    entities: List[str]


class ReasonScore(BaseModel):
    reason: str
    score: float
