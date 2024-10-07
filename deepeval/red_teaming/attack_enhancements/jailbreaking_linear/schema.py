from typing import Literal
from pydantic import BaseModel


class ImprovementPrompt(BaseModel):
    improvement: str
    prompt: str


class OnTopic(BaseModel):
    response: bool


class Rating(BaseModel):
    number: int


class NonRefusal(BaseModel):
    classification: Literal["Non-refusal", "Refusal"]
