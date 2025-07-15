from typing import Optional
from pydantic import BaseModel, Field


class TaskAndOutcome(BaseModel):
    task: str
    outcome: str


class TaskCompletionVerdict(BaseModel):
    verdict: float
    reason: Optional[str] = Field(default=None)
