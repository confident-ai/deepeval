from typing import List, Optional
from pydantic import BaseModel, Field
from deepeval.metrics.base_metric import YesNo


class Misuses(BaseModel):
    misuses: List[str]


class MisuseVerdict(BaseModel):
    verdict: YesNo
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[MisuseVerdict]


class MisuseScoreReason(BaseModel):
    reason: str
