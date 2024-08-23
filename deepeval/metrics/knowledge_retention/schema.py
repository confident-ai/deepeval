from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field


class Knowledge(BaseModel):
    data: Dict[str, Union[str, List[str]]]


class KnowledgeRetentionVerdict(BaseModel):
    index: int
    verdict: str
    reason: Optional[str] = Field(default=None)
