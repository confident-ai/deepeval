from typing import Literal
from pydantic import BaseModel


class ImprovementPrompt(BaseModel):
    improvement: str
    prompt: str


class OnTopic(BaseModel):
    response: bool


class AttackData(BaseModel):
    generated_question: str
    last_response_summary: str
    rationale_behind_jailbreak: str


class RefusalData(BaseModel):
    value: bool
    rationale: str
    metadata: int


class EvalData(BaseModel):
    value: bool
    description: str
    rationale: str
    metadata: int
