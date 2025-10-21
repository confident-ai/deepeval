from pydantic import BaseModel
from typing import List, Dict, Literal


class Task(BaseModel):
    task: str


class EfficiencyVerdict(BaseModel):
    score: float
    reason: str
