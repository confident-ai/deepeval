from pydantic import BaseModel
from typing import List, Dict, Literal


class QAPair(BaseModel):
    question: str
    response: str


class QAPairs(BaseModel):
    qa_pairs: List[QAPair]


class RelevancyVerdict(BaseModel):
    verdict: Literal["TP", "TN", "FP", "FN"]
    reason: str
