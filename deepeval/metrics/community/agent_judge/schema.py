from typing import List

from pydantic import BaseModel, Field


class AgentMilestones(BaseModel):
    milestones: List[str] = Field(
        description=(
            "2-6 ordered, independently checkable milestones derived "
            "from the task."
        )
    )


class AgentJudgment(BaseModel):
    scores: List[float] = Field(
        description=(
            "One 0.0-1.0 verdict per milestone, in milestone order. "
            "Use 1.0 for fully met, 0.5 for partially met, 0.0 for "
            "not met."
        )
    )
    reason: str = Field(
        description="One sentence on which milestones are met or missed."
    )
