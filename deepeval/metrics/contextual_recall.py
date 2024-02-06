from typing import Optional, List, Union
from pydantic import BaseModel, Field
import json
from langchain_core.language_models import BaseChatModel

from deepeval.utils import trimToJson
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseModel
from deepeval.metrics.templates import ContextualRecallTemplate
from deepeval.progress_context import metrics_progress_context


class ContextualRecallVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class ContextualRecallMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseModel, BaseChatModel]] = None,
        include_reason: bool = True,
    ):
        self.threshold = threshold
        if isinstance(model, DeepEvalBaseModel):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
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
        with metrics_progress_context(self.__name__, self.evaluation_model):
            self.verdicts: List[
                ContextualRecallVerdict
            ] = self._generate_verdicts(
                test_case.expected_output, test_case.retrieval_context
            )

            contextual_recall_score = self._generate_score()

            self.reason = self._generate_reason(
                test_case.expected_output, contextual_recall_score
            )

            self.success = contextual_recall_score >= self.threshold
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

        res = self.model(prompt)
        return res

    def _generate_score(self):
        if len(self.verdicts) == 0:
            return 0

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
        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)
        verdicts = [
            ContextualRecallVerdict(**item) for item in data["verdicts"]
        ]

        return verdicts

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Contextual Recall"
