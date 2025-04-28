from pydantic import BaseModel


class Purpose(BaseModel):
    purpose: str


class ReasonScore(BaseModel):
    reason: str
    score: float
