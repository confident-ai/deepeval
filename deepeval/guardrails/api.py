from typing import Optional, List, Union, Dict
from pydantic import BaseModel

BASE_URL = "https://deepeval.confident-ai.com/"


class ApiGuardrails(BaseModel):
    guard: str
    guard_type: str
    vulnerability_types: Optional[list[str]] = None
    input: Optional[str] = None
    response: Optional[str] = None
    purpose: Optional[str] = None
    allowed_topics: Optional[List[str]] = None


class GuardScore(BaseModel):
    guard: str
    score: int
    score_breakdown: Union[List, Dict]


class GuardResponseData(BaseModel):
    result: GuardScore


# Models for running multiple guards


class ApiMultipleGuardrails(BaseModel):
    guard_params: List[ApiGuardrails]


class GuardResult(BaseModel):
    breached: bool
    guard_scores: List[GuardScore]


class GuardsResponseData(BaseModel):
    result: GuardResult
