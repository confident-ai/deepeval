from pydantic import BaseModel, Field


class ReasonScore(BaseModel):
    reasoning: str
    score: float = Field(ge=0, le=10)
