from typing import List, Dict, Union
from pydantic import BaseModel, Field


class Knowledge(BaseModel):
    data: Dict[str, Union[str, List[str]]]


class KnowledgeRetentionVerdict(BaseModel):
    index: int
    verdict: str
    reason: str = Field(default=None)
