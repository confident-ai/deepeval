from typing import Optional, List
from pydantic import BaseModel


class ApiGuardrails(BaseModel):
    input: str
    response: str
    guards: List[str]
    purpose: Optional[str] = None
    allowed_entities: Optional[List[str]] = None
    system_prompt: Optional[str] = None


class GuardScore(BaseModel):
    guard: str
    score: int


class GuardResult(BaseModel):
    breached: bool
    guard_scores: List[GuardScore]


class GuardResponseData(BaseModel):
    result: GuardResult
