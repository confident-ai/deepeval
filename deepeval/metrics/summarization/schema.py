from pydantic import BaseModel, Field
from deepeval.metrics.base_metric import YesNoBorderline
from typing import List, Optional
from enum import Enum


class ScoreType(Enum):
    ALIGNMENT = "Alignment"
    COVERAGE = "Coverage"


class SummarizationAlignmentVerdict(BaseModel):
    verdict: YesNoBorderline
    reason: Optional[str] = Field(default=None)


class SummarizationCoverageVerdict(BaseModel):
    summary_verdict: str
    original_verdict: str
    question: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[SummarizationAlignmentVerdict]


class Questions(BaseModel):
    questions: List[str]


class Answers(BaseModel):
    answers: List[str]


class SummarizationScoreReason(BaseModel):
    reason: str
