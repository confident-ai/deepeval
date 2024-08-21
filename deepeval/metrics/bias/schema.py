from typing import List, Optional
from pydantic import BaseModel, Field


class Opinions(BaseModel):
    opinions: List[str]


# BiasMetric runs a similar algorithm to Dbias: https://arxiv.org/pdf/2208.05777.pdf
class BiasVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[BiasVerdict]


class Reason(BaseModel):
    reason: str
