from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class StepConsistencyVerdict(BaseModel):
    step_index: int = Field(
        ...,
        description="1-based index of the tool call this verdict applies to",
    )
    verdict: Literal["consistent", "inconsistent"]
    reasoning: Optional[str] = Field(default=None)


class ThoughtActionConsistencyVerdict(BaseModel):
    verdicts: List[StepConsistencyVerdict] = Field(
        ..., description="One verdict per judged step, in step order"
    )
