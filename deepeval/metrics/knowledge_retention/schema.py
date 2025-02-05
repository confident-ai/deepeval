from typing import Dict, Optional, Any
from pydantic import BaseModel


class Knowledge(BaseModel):
    data: Dict[str, Any]


class KnowledgeRetentionVerdict(BaseModel):
    verdict: str
    index: Optional[int] = None
    reason: Optional[str] = None


class Reason(BaseModel):
    reason: str
