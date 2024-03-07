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
        run_async: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.run_async = run_async
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase) -> float:
        if test_case.input is None or test_case.actual_output is None:
            raise ValueError("Input or actual output cannot be None")
        with metrics_progress_context(
            self.__name__, self.evaluation_model, self.strict_mode
        ):
            if self.run_async:
                loop = get_or_create_event_loop()
                self.statements: List[str] = loop.run_until_complete(
                    self._a_generate_statements(test_case.actual_output)
                )
                self.verdicts: List[AnswerRelvancyVerdict] = (
                    loop.run_until_complete(
                        self._a_generate_verdicts(test_case.input)
                    )
                )
                self.score = self._generate_score()
                self.reason = loop.run_until_complete(
                    self._a_generate_reason(test_case.input)
                )
            else:
                self.statements: List[str] = self._generate_statements(
                    test_case.actual_output
                )
                self.verdicts: List[AnswerRelvancyVerdict] = (
                    self._generate_verdicts(test_case.input)
                )
                self.score = self._generate_score()
                self.reason = self._generate_reason(test_case.input)

            self.success = self.score >= self.threshold
            capture_metric_type(self.__name__)
            return self.score

    def _generate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        score = relevant_count / number_of_verdicts

        return 0 if self.strict_mode and score < self.threshold else score

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

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        return res

    def _generate_reason(self, input: str) -> str:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self._a_generate_reason(input))

    async def _a_generate_verdicts(
        self, input: str
    ) -> List[AnswerRelvancyVerdict]:
        print("generating verdicts")
        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)
        data = trimAndLoadJson(res)
        verdicts = [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]
        return verdicts

    def _generate_verdicts(self, input: str) -> List[AnswerRelvancyVerdict]:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self._a_generate_verdicts(input))

    async def _a_generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        print("generating statements")
        prompt = AnswerRelevancyTemplate.generate_statements(
            actual_output=actual_output,
        )

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)
        data = trimAndLoadJson(res)
        return data["statements"]

    def _generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self._a_generate_statements(actual_output)
        )

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
