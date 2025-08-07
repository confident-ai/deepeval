from pydantic import BaseModel


class ToolScore(BaseModel):
    score: float
    reason: str
