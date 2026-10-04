from pydantic import BaseModel, Field


class SufficiencyJudgment(BaseModel):
    score: float = Field(
        description="0.0 (missing everything) to 1.0 (fully answerable)."
    )
    reason: str = Field(description="One sentence on what is covered.")
