from typing import List, Literal
from pydantic import BaseModel


class RewrittenInput(BaseModel):
    rewritten_input: str


class InputFeedback(BaseModel):
    score: float
    feedback: str


class SyntheticData(BaseModel):
    input: str


class SyntheticDataList(BaseModel):
    data: List[SyntheticData]


class SQLData(BaseModel):
    sql: str


class ComplianceData(BaseModel):
    non_compliant: bool


class Response(BaseModel):
    response: str


class ImprovementPrompt(BaseModel):
    improvement: str
    prompt: str


class OnTopic(BaseModel):
    response: bool


class Rating(BaseModel):
    number: int


class TreeScore(BaseModel):
    answer_1: int
    answer_2: int
    answer_3: int


class NonRefusal(BaseModel):
    classification: Literal["Non-refusal", "Refusal"]
