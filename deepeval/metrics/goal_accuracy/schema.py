from pydantic import BaseModel
from typing import List


class GoalSteps(BaseModel):
    user_goal: str
    steps_taken: List[str]


class GoalScore(BaseModel):
    score: float
    reason: str


class PlanScore(BaseModel):
    score: float
    reason: str
