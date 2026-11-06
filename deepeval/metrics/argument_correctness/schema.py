from typing import List, Optional
from pydantic import BaseModel, Field
from deepeval.metrics.base_metric import YesNo


class ArgumentCorrectnessVerdict(BaseModel):
    verdict: YesNo
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ArgumentCorrectnessVerdict]


class ArgumentCorrectnessScoreReason(BaseModel):
    reason: str
