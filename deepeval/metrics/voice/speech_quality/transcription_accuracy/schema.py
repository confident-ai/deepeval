from typing import List

from pydantic import BaseModel

from deepeval.metrics.base_metric import YesNo


class TranscriptionAccuracyVerdict(BaseModel):
    verdict: YesNo
    reason: str


class Verdicts(BaseModel):
    verdicts: List[TranscriptionAccuracyVerdict]


class TranscriptionAccuracyScoreReason(BaseModel):
    reason: str
