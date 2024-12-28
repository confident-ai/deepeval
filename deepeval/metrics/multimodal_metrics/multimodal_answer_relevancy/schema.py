from typing import List, Optional
from pydantic import BaseModel, Field


class Statements(BaseModel):
    statements: List[str]


class AnswerRelvancyVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[AnswerRelvancyVerdict]


class Reason(BaseModel):
    reason: str
