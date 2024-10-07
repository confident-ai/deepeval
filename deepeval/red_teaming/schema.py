from typing import List, Literal
from pydantic import BaseModel


class RewrittenInput(BaseModel):
    rewritten_input: str


class SyntheticData(BaseModel):
    input: str


class SyntheticDataList(BaseModel):
    data: List[SyntheticData]


class ComplianceData(BaseModel):
    non_compliant: bool


class ImprovementPrompt(BaseModel):
    improvement: str
    prompt: str


class OnTopic(BaseModel):
    response: bool


class Rating(BaseModel):
    number: int


class NonRefusal(BaseModel):
    classification: Literal["Non-refusal", "Refusal"]
