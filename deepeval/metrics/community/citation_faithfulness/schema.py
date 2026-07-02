from typing import Optional, Literal
from pydantic import BaseModel, Field


class CitationFaithfulnessVerdict(BaseModel):
    verdict: Literal["faithful", "unfaithful"]
    reasoning: Optional[str] = Field(default=None)
