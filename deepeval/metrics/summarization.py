from typing import List, Optional
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class SummarizationMetric(BaseMetric):
    def __init__(
        self,
        minimum_score: float = 0.5,
        model: Optional[str] = None,
        n: Optional[int] = 5,
        assessment_questions: Optional[List[str]] = None,
        azure_deployment_name: Optional[str] = None,
    ):
        self.minimum_score = minimum_score
        self.model = model
        self.assessment_questions = assessment_questions
        self.azure_deployment_name = azure_deployment_name
        self.n = n
        self.alignment_score = None
        self.inclusion_score = None

    def measure(self, test_case: LLMTestCase):
        if test_case.input is None or test_case.actual_output is None:
            raise ValueError("Input or actual output cannot be None")

        source_document = test_case.input
        summary = test_case.actual_output

        with ThreadPoolExecutor() as executor:
            future_alignment = executor.submit(
                self.get_score, ScoreType.ALIGNMENT, source_document, summary
            )
            future_inclusion = executor.submit(
                self.get_score, ScoreType.INCLUSION, source_document, summary
            )

            # Wait for the results
            alignment_score = future_alignment.result()
            inclusion_score = future_inclusion.result()

        summarization_score = min(alignment_score, inclusion_score)

        self.success = summarization_score >= self.minimum_score
        self.score_metadata = {
            "Alignment": alignment_score,
            "Inclusion": inclusion_score,
        }
        self.alignment_score = alignment_score
        self.inclusion_score = inclusion_score
        self.score = summarization_score
        return self.score

    def get_score(
        self, score_type: ScoreType, source_document: str, summary: str
    ):
        questions = []
        if score_type == ScoreType.ALIGNMENT:
            print("Calculating alignment score...")
            questions = self.generate_questions(
                score_type, source_document, summary
            )
        elif score_type == ScoreType.INCLUSION:
            print("Calculating inclusion score...")
            if self.assessment_questions is None:
                questions = self.generate_questions(
                    score_type, source_document, summary
                )
            else:
                questions = self.assessment_questions

        score = 0
        interval = 1 / len(questions)
        for question in questions:
            with ThreadPoolExecutor() as executor:
                future_source_answer = executor.submit(
                    self.get_answer, question, source_document
                )
                future_summary_answer = executor.submit(
                    self.get_answer, question, summary
                )
                source_answer = future_source_answer.result()
                summary_answer = future_summary_answer.result()

            if source_answer.strip().lower() == summary_answer.strip().lower():
                score += interval

        return score

    def generate_questions(
        self,
        score_type: ScoreType,
        source_document: str,
        summary: str,
    ) -> List[str]:
        if score_type == ScoreType.ALIGNMENT:
            prompt: dict = closed_end_questions_template.format(
                n=self.n, text=summary
            )
        elif score_type == ScoreType.INCLUSION:
            prompt: dict = closed_end_questions_template.format(
                n=self.n, text=source_document
            )

        model_kwargs = {}
        if self.azure_deployment_name is not None:
            model_kwargs["deployment_id"] = self.azure_deployment_name

        chat_model = GPTModel(model_name=self.model, model_kwargs=model_kwargs)
        res = chat_model(prompt)

        json_output = trimToJson(res.content)
        data = json.loads(json_output)

        return data["questions"]

    def get_answer(self, question: str, text: str) -> str:
        prompt: dict = closed_end_answers_template.format(
            question=question, text=text
        )

        model_kwargs = {}
        if self.azure_deployment_name is not None:
            model_kwargs["deployment_id"] = self.azure_deployment_name

        chat_model = GPTModel(model_name=self.model, model_kwargs=model_kwargs)
        res = chat_model(prompt)

        return res.content

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Summarization"
