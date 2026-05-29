from pydantic import BaseModel
from typing import List, Dict, Literal, Optional


class Task(BaseModel):
    task: str


class StepAnalysis(BaseModel):
    step_name: str
    is_necessary: bool
    reason: str


class EfficiencyVerdict(BaseModel):
    score: float
    reason: str
    steps: Optional[List[StepAnalysis]] = None
