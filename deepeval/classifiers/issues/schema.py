from pydantic import BaseModel, Field


class IssueReview(BaseModel):
    what_happened: str = ""
    harm: str = ""
    instruction: str = ""
    confidence: float = Field(ge=0, le=1)
    reason: str = ""
