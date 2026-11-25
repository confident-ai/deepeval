from pydantic import BaseModel


class NonRefusal(BaseModel):
    refusal: bool
    reasoning: str
