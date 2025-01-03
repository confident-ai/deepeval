from typing import Optional, List, Union, Dict
from pydantic import BaseModel

BASE_URL = "http://localhost:8000"

class ApiGuardrails(BaseModel):
    guard: str
    guard_type: str
    vulnerability_types: Optional[list[str]] = None
    input: Optional[str] = None
    response: Optional[str] = None
    purpose: Optional[str] = None
    allowed_topics: Optional[List[str]] = None

class GuardResult(BaseModel):
    breached: bool
    result_breakdown: Union[List, Dict]

class GuardResponseData(BaseModel):
    result: GuardResult
