from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class CitationFaithfulnessVerdict(BaseModel):
    verdict: Literal["faithful", "unfaithful"]
    reasoning: Optional[str] = Field(default=None)

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict(cls, value: Any) -> Any:
        # Judges often reply "Faithful"; fold case before the Literal check.
        return value.strip().lower() if isinstance(value, str) else value
