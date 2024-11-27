from typing import List, Optional
from pydantic import BaseModel, Field


class PromptAlignmentVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[PromptAlignmentVerdict]


class Reason(BaseModel):
    reason: str
