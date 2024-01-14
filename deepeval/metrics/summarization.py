from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field
import json

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel
from deepeval.utils import trimToJson
from deepeval.templates import (
    closed_end_questions_template,
    closed_end_answers_template,
)


class ScoreType(Enum):
    INCLUSION = "inclusion"
    ALIGNMENT = "alignment"


class Verdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class SummarizationVerdict(BaseModel):
    question: str = Field(default=None)
    original_text: Verdict
    summary: Verdict


class SummarizationMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[str] = None,
        n: Optional[int] = 5,
        assessment_questions: Optional[List[str]] = None,
        include_reason: bool = False,
    ):
        self.threshold = threshold
        self.model = model
        self.inclusion_questions = assessment_questions
        self.n = n
        self.alignment_score = None
        self.inclusion_score = None
        self.include_reason = include_reason

    def measure(self, test_case: LLMTestCase):
        if test_case.input is None or test_case.actual_output is None:
            raise ValueError("Input or actual output cannot be None")

        original_text = test_case.input
        summary = test_case.actual_output

        alignment_score = self._generate_score(
            ScoreType.ALIGNMENT, original_text, summary
        )
        inclusion_score = self._generate_score(
            ScoreType.INCLUSION, original_text, summary
        )

        summarization_score = min(alignment_score, inclusion_score)

        self.reason = self._generate_reason(summarization_score)

        self.success = summarization_score >= self.threshold
        self.score_breakdown = {
            "Alignment": alignment_score,
            "Inclusion": inclusion_score,
        }
        self.alignment_score = alignment_score
        self.inclusion_score = inclusion_score
        self.score = summarization_score
        return self.score

    def _generate_reason(self, score: float):
        if self.include_reason:
            # TODO: construct verdicts and generate reason for both alignment and inclusion, pass in as json
            pass
        else:
            return None

    def _generate_score(
        self, score_type: ScoreType, original_text: str, summary: str
    ):
        if score_type == ScoreType.ALIGNMENT:
            self.alignment_questions = self._generate_questions(
                score_type, original_text, summary
            )
            self.alignment_verdicts = self._generate_verdicts(
                score_type, original_text, summary
            )
            # generate score for each verdict using subverdicts
            count = 0
            for verdict in self.alignment_verdicts:
                if (
                    verdict.original_text.verdict.strip().lower()
                    == verdict.summary.verdict.strip().lower()
                ):
                    count += 1
            return count / len(self.alignment_verdicts)

        elif score_type == ScoreType.INCLUSION:
            if self.inclusion_questions is None:
                self.inclusion_questions = self._generate_questions(
                    score_type, original_text, summary
                )
            self.inclusion_verdicts = self._generate_verdicts(
                score_type, original_text, summary
            )
            count = 0
            for verdict in self.inclusion_verdicts:
                if (
                    verdict.original_text.verdict.strip().lower()
                    == verdict.summary.verdict.strip().lower()
                ):
                    count += 1
            return count / len(self.inclusion_verdicts)
            # generate score for each verdict using subverdicts

    def _generate_questions(
        self, score_type: ScoreType, original_text: str, summary: str
    ):
        if score_type == ScoreType.ALIGNMENT:
            reference_text = summary
        elif score_type == ScoreType.INCLUSION:
            reference_text = original_text

        # TODO: add generate questions template

        pass

    def _generate_verdicts(
        self, score_type: ScoreType, original_text: str, summary: str
    ) -> List[SummarizationVerdict]:
        if score_type == ScoreType.ALIGNMENT:
            questions = self.alignment_questions
        elif score_type == ScoreType.INCLUSION:
            questions = self.inclusion_questions

        # TODO: generate verdicts based on questions

        pass

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Summarization"
