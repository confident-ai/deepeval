from typing import List
from pydantic import BaseModel, Field


class ReasonScore(BaseModel):
    reason: str
    score: float = Field(ge=0, le=10)


class BestTestCase(BaseModel):
    best_test_case_index: int = None
    best_test_case_id: str = None
    reason: str


class Steps(BaseModel):
    steps: List[str]
