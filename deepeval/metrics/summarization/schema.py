from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class ScoreType(Enum):
    ALIGNMENT = "Alignment"
    COVERAGE = "Coverage"


class SummarizationAlignmentVerdict(BaseModel):
    # yes, no, or idk
    verdict: str
    reason: str = Field(default=None)


class SummarizationCoverageVerdict(BaseModel):
    summary_verdict: str
    original_verdict: str
    question: str = Field(default=None)


class Verdict(BaseModel):
    verdicts: List[SummarizationAlignmentVerdict]


class Questions(BaseModel):
    questions: List[str]


class Answers(BaseModel):
    answers: List[str]


class Answers(BaseModel):
    answers: List[str]


class Reason(BaseModel):
    reason: str
