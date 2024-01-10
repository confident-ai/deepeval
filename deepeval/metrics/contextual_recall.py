from typing import Optional, List
from pydantic import BaseModel, Field
import json

from deepeval.utils import trimToJson
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel
from deepeval.templates import ContextualRecallTemplate
from deepeval.progress_context import metrics_progress_context


class ContextualRecallVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class ContextualRecallMetric(BaseMetric):
    def __init__(
        self,
        minimum_score: float = 0.5,
        model: Optional[str] = None,
        include_reason: bool = True,
    ):
        self.minimum_score = minimum_score
        self.model = model
        self.include_reason = include_reason
        self.n = 5

    def measure(self, test_case: LLMTestCase) -> float:
        if (
            test_case.input is None
            or test_case.actual_output is None
            or test_case.retrieval_context is None
            or test_case.expected_output is None
        ):
            raise ValueError(
                "Input, actual output, expected output, or retrieval context cannot be None"
            )
        with metrics_progress_context(self.__name__):
            self.verdicts: List[
                ContextualRecallVerdict
            ] = self._generate_verdicts(
                test_case.expected_output, test_case.retrieval_context
            )

            contextual_recall_score = self._generate_score()

            self.reason = self._generate_reason(
                test_case.expected_output, contextual_recall_score
            )

            self.success = contextual_recall_score >= self.minimum_score
            self.score = contextual_recall_score
            return self.score

    def _generate_reason(self, expected_output: str, score: float):
        if self.include_reason is False:
            return None

        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = ContextualRecallTemplate.generate_reason(
            expected_output=expected_output,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
            score=format(score, ".2f"),
        )
        chat_model = GPTModel(model_name=self.model)
        res = chat_model(prompt)

        return res.content

    def _generate_score(self):
        justified_sentences = 0
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                justified_sentences += 1

        return justified_sentences / len(self.verdicts)

    def _generate_verdicts(
        self, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualRecallVerdict]:
        prompt = ContextualRecallTemplate.generate_verdicts(
            expected_output=expected_output, retrieval_context=retrieval_context
        )
        chat_model = GPTModel(model_name=self.model)
        res = chat_model(prompt)
        json_output = trimToJson(res.content)
        data = json.loads(json_output)
        verdicts = [
            ContextualRecallVerdict(**item) for item in data["verdicts"]
        ]

        return verdicts

    def is_successful(self) -> bool:
        self.success = self.score >= self.minimum_score
        return self.success

    @property
    def __name__(self):
        return "Contextual Recall"
