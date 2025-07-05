from typing import List
from pydantic import BaseModel


class NonAdviceVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[NonAdviceVerdict]


class AdviceStatements(BaseModel):
    advice_statements: List[str]


class Reason(BaseModel):
    reason: str 