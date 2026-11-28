from pydantic import BaseModel, Field
from typing import List


class Entities(BaseModel):
    entities: List[str]


class ReasonScore(BaseModel):
    reason: str
    score: float
