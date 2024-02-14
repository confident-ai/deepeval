from typing import Optional, List, Union
from pydantic import BaseModel, Field
import json

from deepeval.utils import trimToJson
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseModel
from deepeval.metrics.templates import AnswerRelevancyTemplate
from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type


class AnswerRelvancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class AnswerRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseModel]] = None,
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
        ):
            raise ValueError(
                "Input, actual output, or retrieval context cannot be None"
            )
        with metrics_progress_context(self.__name__, self.evaluation_model):
            # generate statements
            self.statements: List[str] = self._generate_statements(
                test_case.actual_output
            )

            # generate verdicts based on statements, and retrieval context
            self.verdicts: List[AnswerRelvancyVerdict] = (
                self._generate_verdicts(
                    test_case.input, test_case.retrieval_context
                )
            )

            answer_relevancy_score = self._generate_score()

            self.reason = self._generate_reason(
                test_case.input, answer_relevancy_score
            )
            self.success = answer_relevancy_score >= self.threshold
            self.score = answer_relevancy_score
            capture_metric_type(self.__name__)
            return self.score

    def _generate_score(self):
        if len(self.verdicts) == 0:
            return 0

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        return relevant_count / len(self.verdicts)

    def _generate_reason(self, input: str, score: float) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(score, ".2f"),
        )

        res = self.model(prompt)
        return res

    def _generate_verdicts(
        self, input: str, retrieval_context=List[str]
    ) -> List[AnswerRelvancyVerdict]:
        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
            retrieval_context="\n\n".join(retrieval_context),
        )

        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)
        verdicts = [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]
        return verdicts

    def _generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_statements(
            actual_output=actual_output,
        )

        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)
        return data["statements"]

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
