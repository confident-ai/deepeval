from typing import Literal
from pydantic import BaseModel


class ImprovementPrompt(BaseModel):
    improvement: str
    prompt: str


class OnTopic(BaseModel):
    on_topic: bool


class Rating(BaseModel):
    rating: int


class NonRefusal(BaseModel):
    classification: Literal["Non-refusal", "Refusal"]
