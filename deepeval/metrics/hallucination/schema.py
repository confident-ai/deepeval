from typing import List, Literal
from pydantic import BaseModel


class HallucinationVerdict(BaseModel):
    verdict: Literal["yes", "no"]
    reason: str


class Verdicts(BaseModel):
    verdicts: List[HallucinationVerdict]


class Reason(BaseModel):
    reason: str
