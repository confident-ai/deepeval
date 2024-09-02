from pydantic import BaseModel
from typing import List


class Purpose(BaseModel):
    purpose: str


class Entities(BaseModel):
    entities: List[str]


class ReasonScore(BaseModel):
    reason: str
    score: float
