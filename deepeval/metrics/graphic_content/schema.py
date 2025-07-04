from typing import List
from pydantic import BaseModel


class GraphicContentVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[GraphicContentVerdict]


class Reason(BaseModel):
    reason: str 