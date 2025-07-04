from typing import List
from pydantic import BaseModel


class IllegalActivityVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[IllegalActivityVerdict]


class Reason(BaseModel):
    reason: str 