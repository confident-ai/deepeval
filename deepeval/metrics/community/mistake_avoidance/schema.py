from typing import List
from pydantic import BaseModel
from deepeval.metrics.base_metric import YesNo


class MistakeAvoidanceVerdict(BaseModel):
    verdict: YesNo
    reason: str


class Verdicts(BaseModel):
    verdicts: List[MistakeAvoidanceVerdict]
