from pydantic import BaseModel, Field


class Purpose(BaseModel):
    purpose: str


class ReasonScore(BaseModel):
    reason: str
    score: float
