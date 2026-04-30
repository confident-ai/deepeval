from typing import List
from pydantic import BaseModel


class ScorerDiagnosisSchema(BaseModel):
    failures: str
    successes: str
    analysis: str


class ScorerDiagnosisResult(BaseModel):
    failures: str
    successes: str
    analysis: str
    results: List[str]
