from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class NamedReference(BaseModel):
    """A mention in `actual_output` of a document-native structural label
    (e.g. "Table 3", "Section 4.2", "footnote 4") together with the claim
    attached to it."""

    label: str
    claim: str


class NamedReferences(BaseModel):
    references: List[NamedReference]


class NamedReferenceVerdict(BaseModel):
    label: str
    verdict: Literal["yes", "no", "idk"]
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[NamedReferenceVerdict]


class NamedReferenceAttributionScoreReason(BaseModel):
    reason: str
