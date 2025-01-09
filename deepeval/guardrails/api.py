from typing import Optional, List, Union, Dict
from pydantic import BaseModel

BASE_URL = "https://deepeval.confident-ai.com/"


class ApiGuard(BaseModel):
    guard: str
    guard_type: str
    vulnerability_types: Optional[list[str]] = None
    input: Optional[str] = None
    response: Optional[str] = None
    purpose: Optional[str] = None
    allowed_topics: Optional[List[str]] = None


class GuardData(BaseModel):
    guard: str
    score: int
    reason: str
    score_breakdown: Union[List, Dict]


# Models for running multiple guards


class ApiGuardrails(BaseModel):
    guards: List[ApiGuard]


class GuardResult(BaseModel):
    breached: bool
    guard_data: List[GuardData]


class GuardsResponseData(BaseModel):
    result: GuardResult
