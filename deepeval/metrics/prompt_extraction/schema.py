from typing import List
from pydantic import BaseModel


class PromptExtractionVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[PromptExtractionVerdict]


class Reason(BaseModel):
    reason: str 