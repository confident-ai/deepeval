from typing import List, Optional
from pydantic import BaseModel, Field


class GoalAndOutcome(BaseModel):
    user_goal: str
    task_outcome: str


class TaskCompletionVerdict(BaseModel):
    verdict: float
    reason: Optional[str] = Field(default=None)
