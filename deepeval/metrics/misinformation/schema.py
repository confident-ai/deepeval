from typing import List
from pydantic import BaseModel


class MisinformationVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[MisinformationVerdict]


class Reason(BaseModel):
    reason: str 