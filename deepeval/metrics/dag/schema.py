from typing import Literal, Dict, Union
from pydantic import BaseModel


class Reason(BaseModel):
    reason: str


class TaskNodeOutput(BaseModel):
    output: Union[str, list, Dict]


class BinaryJudgementVerdict(BaseModel):
    verdict: Literal[True, False]
    reason: str


class NonBinaryJudgementVerdict(BaseModel):
    verdict: str
    reason: str
