from pydantic import BaseModel
from typing import List


class Task(BaseModel):
    task: str
    steps_taken: List[str]


class TaskScore(BaseModel):
    score: float
    reason: str


class ToolScore(BaseModel):
    score: float
    reason: str


class ArgsScore(BaseModel):
    score: float
    reason: str
