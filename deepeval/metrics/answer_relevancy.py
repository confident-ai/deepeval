from typing import Optional, List, Union
from pydantic import BaseModel, Field
import json
from langchain_core.language_models import BaseChatModel

from deepeval.utils import trimToJson
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseModel
from deepeval.metrics.templates import AnswerRelevancyTemplate
from deepeval.progress_context import metrics_progress_context


class AnswerRelvancyVerdict(BaseModel):
    verdict: str
    key_point: str = Field(default=None)


class AnswerRelevancyMetric(BaseMetric):
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
        ):
            raise ValueError(
                "Input, actual output, or retrieval context cannot be None"
            )
        with metrics_progress_context(self.__name__, self.evaluation_model):
            self.key_points: List[str] = self._generate_key_points(
                test_case.actual_output, "\n".join(test_case.retrieval_context)
            )
            self.verdicts: List[
                AnswerRelvancyVerdict
            ] = self._generate_verdicts(test_case.input)

            answer_relevancy_score = self._generate_score()

            self.reason = self._generate_reason(
                test_case.input, test_case.actual_output, answer_relevancy_score
            )
            self.success = answer_relevancy_score >= self.threshold
            self.score = answer_relevancy_score
            return self.score

    def _generate_score(self):
        if len(self.verdicts) == 0:
            return 0

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        return relevant_count / len(self.verdicts)

    def _generate_reason(
        self, original_question: str, answer: str, score: float
    ) -> str:
        if self.include_reason is False:
            return None

        irrelevant_points = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_points.append(verdict.key_point)

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_points=irrelevant_points,
            original_question=original_question,
            answer=answer,
            score=format(score, ".2f"),
        )

        res = self.model(prompt)
        return res

    def _generate_verdicts(
        self, original_question: str
    ) -> List[AnswerRelvancyVerdict]:
        prompt = AnswerRelevancyTemplate.generate_verdicts(
            original_question=original_question, key_points=self.key_points
        )

        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)
        verdicts = [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]

        if len(verdicts) != len(self.key_points):
            raise ValueError("Number of verdicts generated does not equal.")

        for i in range(len(verdicts)):
            verdicts[i].key_point = self.key_points[i]

        return verdicts

    def _generate_key_points(
        self, answer: str, retrieval_context: str
    ) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_key_points(
            answer=answer, retrieval_context=retrieval_context
        )

        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)
        return data["key_points"]

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
