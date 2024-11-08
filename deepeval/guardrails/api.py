from typing import Optional, List
from pydantic import BaseModel

class APIGuard(BaseModel):
    input: str
    response: str
    guards: List[str]
    purpose: Optional[str] = None
    allowed_entities: Optional[List[str]] = None
    system_prompt: Optional[str] = None
