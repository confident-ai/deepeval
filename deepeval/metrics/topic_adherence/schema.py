from pydantic import BaseModel, field_validator
from typing import Any, List, Dict, Literal


class QAPair(BaseModel):
    question: str
    response: str


class QAPairs(BaseModel):
    qa_pairs: List[QAPair]


class RelevancyVerdict(BaseModel):
    verdict: Literal["TP", "TN", "FP", "FN"]
    reason: str

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict(cls, value: Any) -> Any:
        # Judges often reply "tp"; fold case before the Literal check.
        return value.strip().upper() if isinstance(value, str) else value


class TopicAdherenceReason(BaseModel):
    reason: str
