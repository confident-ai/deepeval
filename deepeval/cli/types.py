from typing import Optional, List
from pydantic import BaseModel


class RecommendMetricsRequestData(BaseModel):
    questionIndex: int
    userAnswers: Optional[List[bool]]


class RecommendMetricsResponseData(BaseModel):
    isLastQuestion: bool
    question: Optional[str]
    recommendedMetrics: List[str]
