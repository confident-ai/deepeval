from typing import Literal, Dict, Union
from pydantic import BaseModel


class ConversationalMetricScoreReason(BaseModel):
    reason: str


class ConversationalTaskNodeOutput(BaseModel):
    output: Union[str, list[str], dict[str, str]]


class ConversationalBinaryJudgementVerdict(BaseModel):
    verdict: Literal[True, False]
    reason: str


class ConversationalNonBinaryJudgementVerdict(BaseModel):
    verdict: str
    reason: str
