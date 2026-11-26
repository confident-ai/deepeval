from typing import Annotated, List
from pydantic import BaseModel, Field


class ReasonScore(BaseModel):
    reasoning: str
    score: List[Annotated[float, Field(ge=0, le=10)]] = Field(min_length=1)
