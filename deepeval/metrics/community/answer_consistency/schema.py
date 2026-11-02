from typing import Optional

from pydantic import BaseModel, Field


class AgreementScores(BaseModel):
    """Each dimension is rated 1 (completely different) to 5 (identical)."""

    core_facts: int
    specifics: int
    supporting_detail: int
    stance: int
    format: int


class AgreementVerdict(BaseModel):
    reason: Optional[str] = Field(default=None)
    scores: AgreementScores
