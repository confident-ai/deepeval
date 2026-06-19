from pydantic import BaseModel

class Task(BaseModel):
    task: str

class LoopVerdict(BaseModel):
    score: float
    reason: str
