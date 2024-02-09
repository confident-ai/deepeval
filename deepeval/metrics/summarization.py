from typing import List, Optional, Union
from enum import Enum
import json
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseModel
from deepeval.utils import trimToJson
from deepeval.metrics.templates import (
    FaithfulnessTemplate,
    SummarizationTemplate,
)

from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type


class SummarizationAlignmentVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)

class SummarizationInclusionVerdict(BaseModel):
    summary_verdict: str
    original_verdict: str
    question: str = Field(default=None)

class ScoreType(Enum):
    INCLUSION = "Inclusion"
    ALIGNMENT = "Alignment"

class SummarizationMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseModel]] = None,
        n: Optional[int] = 5,
        assessment_questions: Optional[List[str]] = None,
        multithreading=True,
    ):
        self.threshold = threshold
        if isinstance(model, DeepEvalBaseModel):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.assessment_questions = assessment_questions
        self.n = n
        self.alignment_score = None
        self.inclusion_score = None
        self.multithreading = multithreading

    def measure(self, test_case: LLMTestCase):
        if test_case.input is None or test_case.actual_output is None:
            raise ValueError("Input or actual output cannot be None")

        with metrics_progress_context(self.__name__, self.evaluation_model):
            if test_case.input is None or test_case.actual_output is None:
                raise ValueError("Input and actual output cannot be None")

            if self.multithreading:
                # Use multithreading to generate truths and claims in parallel
                with ThreadPoolExecutor() as executor:
                    future_truths = executor.submit(
                        # Claims made in the original text === truths
                        self._generate_claims,
                        test_case.input,
                    )
                    future_claims = executor.submit(
                        self._generate_claims, test_case.actual_output
                    )
                    self.truths: List[str] = future_truths.result()
                    self.claims: List[str] = future_claims.result()
            else:
                # Sequential execution
                self.truths: List[str] = self._generate_claims(test_case.input)
                self.claims: List[str] = self._generate_claims(
                    test_case.actual_output
                )

            self.alignment_verdicts: List[SummarizationAlignmentVerdict] = (
                self._generate_alignment_verdicts(ScoreType.ALIGNMENT)
            )
            alignment_score = self._generate_score(ScoreType.ALIGNMENT)

            if self.assessment_questions:
                self.inclusion_verdicts: List[SummarizationAlignmentVerdict] = (
                    self._generate_verdicts
                )
            inclusion_score = self._generate_score(ScoreType.INCLUSION)

            summarization_score = min(alignment_score, inclusion_score)

            self.success = summarization_score >= self.threshold
            self.score_breakdown = {
                ScoreType.ALIGNMENT.value: alignment_score,
                ScoreType.INCLUSION.value: inclusion_score,
            }
            self.alignment_score = alignment_score
            self.inclusion_score = inclusion_score
            self.score = summarization_score
            capture_metric_type(self.__name__)
            return self.score
    
    def _generate_answers(self) -> List[str]:
        
    
    def _generate_inclusion_verdicts(self, test_case: LLMTestCase) -> List[SummarizationInclusionVerdict]:
        if self.multithreading:
            with ThreadPoolExecutor() as executor:
                future_original_answers: List[str] = executor.submit(
                    self._generate_answers, test_case.input
                )
                future_summary_answers: List[str] = executor.submit(
                    self._generate_answers, test_case.actual_output
                )
                original_answers = future_original_answers.result()
                summary_answers = future_summary_answers.result()
        
        else:
            original_answers = self._generate_answers(test_case.input)
            summary_answers = self._generate_answers(test_case.actual_output)

        if len(original_answers) != len(summary_answers):
            raise ValueError("Number of verdicts generated does not equal.")

        inclusion_veridcts : List[SummarizationInclusionVerdict]= []
        for i in range(len(original_answers)):
            inclusion_veridcts.append(SummarizationInclusionVerdict(summary_verdict=summary_answers[i], original_verdict=original_answers[i], question=self.assessment_questions[i]))

        return inclusion_veridcts
        



    def _generate_alignment_verdicts(
        self, score_type: ScoreType
    ) -> List[SummarizationAlignmentVerdict]:
        verdicts: List[SummarizationAlignmentVerdict] = []
        if score_type == ScoreType.ALIGNMENT:
            prompt = SummarizationTemplate.generate_alignment_verdicts(
                actual_output=self.claims, input="\n\n".join(self.truths)
            )
        else:
            prompt = SummarizationTemplate.generate_inclusion_verdicts(
                actual_output=self.claims, input="\n\n".join(self.truths)
            )
        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)
        verdicts = [SummarizationAlignmentVerdict(**item) for item in data["verdicts"]]

        return verdicts

    def _generate_claims(self, text: str) -> List[str]:
        # Borrow faithfulness template
        prompt = FaithfulnessTemplate.generate_claims(text=text)
        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)

        return data["claims"]

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Summarization"
