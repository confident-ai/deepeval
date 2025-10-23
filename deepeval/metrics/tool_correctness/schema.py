from pydantic import BaseModel


class ToolSelectionScore(BaseModel):
    score: float
    reason: str
