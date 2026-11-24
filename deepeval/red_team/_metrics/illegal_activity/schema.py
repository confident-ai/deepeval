from pydantic import BaseModel, Field


class ReasonScore(BaseModel):
    reason: str
    score: float
