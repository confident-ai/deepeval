from typing import List
from pydantic import BaseModel


class PIILeakageVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[PIILeakageVerdict]


class PIIStatements(BaseModel):
    pii_statements: List[str]


class Reason(BaseModel):
    reason: str 