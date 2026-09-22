from typing import List
from pydantic import BaseModel
from deepeval.metrics.base_metric import YesNo


class ContextualRecallVerdict(BaseModel):
    verdict: YesNo
    reason: str


class VerdictWithExpectedOutput(BaseModel):
    verdict: YesNo
    reason: str
    expected_output: str


class Verdicts(BaseModel):
    verdicts: List[ContextualRecallVerdict]


class ContextualRecallScoreReason(BaseModel):
    reason: str
