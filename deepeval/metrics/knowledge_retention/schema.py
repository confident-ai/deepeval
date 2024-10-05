from typing import List, Dict, Union, Optional
from pydantic import BaseModel


class Knowledge(BaseModel):
    data: Dict[str, Union[str, List[str]]]


class KnowledgeRetentionVerdict(BaseModel):
    verdict: str
    index: Optional[int] = None
    reason: Optional[str] = None


class Reason(BaseModel):
    reason: str
