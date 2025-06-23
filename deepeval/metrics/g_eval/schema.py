from typing import List
from pydantic import BaseModel


class ReasonScore(BaseModel):
    reason: str
    score: float

class BestTestCase(BaseModel):
    best_test_case_index: int
    reason: str

class Steps(BaseModel):
    steps: List[str]
