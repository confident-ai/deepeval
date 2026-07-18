from typing import Optional
from pydantic import BaseModel, Field


class AbstentionVerdict(BaseModel):
    context_supports_answer: bool
    output_abstained: bool
    reasoning: Optional[str] = Field(default=None)
