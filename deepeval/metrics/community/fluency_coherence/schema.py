from pydantic import BaseModel, Field


class FluencyJudgment(BaseModel):
    score: float = Field(description="1 (broken) to 5 (polished).")
    reason: str = Field(description="One sentence on the rating.")
