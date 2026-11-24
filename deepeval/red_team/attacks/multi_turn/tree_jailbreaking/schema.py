from pydantic import BaseModel


class ImprovementPrompt(BaseModel):
    improvement: str
    prompt: str


class OnTopic(BaseModel):
    on_topic: bool


class Rating(BaseModel):
    rating: int
