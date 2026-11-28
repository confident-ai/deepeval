from pydantic import BaseModel


class ReasonScore(BaseModel):
    reason: str
    score: float
