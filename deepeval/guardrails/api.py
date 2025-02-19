from typing import Optional, List, Union, Dict
from pydantic import BaseModel


from deepeval.guardrails.types import GuardType

BASE_URL = "https://deepeval.confident-ai.com/"


class ApiGuard(BaseModel):
    guard: str
    vulnerability_types: Optional[list[str]] = None
    purpose: Optional[str] = None
    allowed_topics: Optional[List[str]] = None


class GuardData(BaseModel):
    guard: str
    score: int
    reason: str
    score_breakdown: Union[List, Dict]


class ApiGuardrails(BaseModel):
    input: str = None
    output: Optional[str] = None
    guards: List[ApiGuard]
    type: GuardType

    class Config:
        use_enum_values = True


class GuardResult(BaseModel):
    breached: bool
    guard_data: List[GuardData]


class GuardsResponseData(BaseModel):
    result: GuardResult
