from pydantic import BaseModel, Field


class ReasonScore(BaseModel):
    reason: str
    score: float


class Steps(BaseModel):
    steps: str
