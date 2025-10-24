from typing import Literal, Dict, Union
from pydantic import BaseModel


class MetricScoreReason(BaseModel):
    reason: str


class TaskNodeOutput(BaseModel):
    output: Union[str, list[str], dict[str, str]]


class BinaryJudgementVerdict(BaseModel):
    verdict: bool
    reason: str


class NonBinaryJudgementVerdict(BaseModel):
    verdict: str
    reason: str
