import asyncio
from typing import Optional, List, Union
from pydantic import BaseModel, Field

from deepeval.utils import trimAndLoadJson, get_or_create_event_loop
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.metrics.answer_relevancy.template import AnswerRelevancyTemplate
from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type


class AnswerRelvancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class AnswerRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        asynchronous: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.asynchronous = asynchronous
        self.strict_mode = strict_mode

    def measure(
        self, test_case: LLMTestCase, _asynchronous: Optional[bool] = None
    ) -> float:
        if test_case.input is None or test_case.actual_output is None:
            raise ValueError("Input or actual output cannot be None")
        asynchronous = (
            _asynchronous if _asynchronous is not None else self.asynchronous
        )
        with metrics_progress_context(
            self.__name__,
            self.evaluation_model,
            self.strict_mode,
            asynchronous,
        ):
            if asynchronous:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.statements: List[str] = self._generate_statements(
                    test_case.actual_output
                )
                self.verdicts: List[AnswerRelvancyVerdict] = (
                    self._generate_verdicts(test_case.input)
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                capture_metric_type(self.__name__)
                return self.score

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ) -> float:
        with metrics_progress_context(
            self.__name__,
            self.evaluation_model,
            self.strict_mode,
            True,
            _show_indicator,
        ):
            self.statements: List[str] = await self._a_generate_statements(
                test_case.actual_output
            )
            self.verdicts: List[AnswerRelvancyVerdict] = (
                await self._a_generate_verdicts(test_case.input)
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(test_case.input)
            self.success = self.score >= self.threshold
            capture_metric_type(self.__name__)
            return self.score

    async def _a_generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )
        res = await self.model.a_generate(prompt)
        return res

    def _generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )
        res = self.model.generate(prompt)
        return res

    async def _a_generate_verdicts(
        self, input: str
    ) -> List[AnswerRelvancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res)
        verdicts = [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]
        return verdicts

    def _generate_verdicts(self, input: str) -> List[AnswerRelvancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        res = self.model.generate(prompt)
        data = trimAndLoadJson(res)
        verdicts = [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]
        return verdicts

    async def _a_generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_statements(
            actual_output=actual_output,
        )
        res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res)
        return data["statements"]

    def _generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_statements(
            actual_output=actual_output,
        )
        res = self.model.generate(prompt)
        data = trimAndLoadJson(res)
        return data["statements"]

    def _calculate_score(self):
        print(self.statements)
        print(self.verdicts)
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        score = relevant_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
