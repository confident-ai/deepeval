from typing import Optional, List, Union
from pydantic import BaseModel, Field

from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    validate_conversational_test_case,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.answer_relevancy.template import AnswerRelevancyTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.answer_relevancy.schema import *

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
]


class AnswerRelvancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class AnswerRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = validate_conversational_test_case(test_case, self)
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self):
            if self.async_mode:
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
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Statements:\n{prettify_list(self.statements)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score

    async def a_measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = validate_conversational_test_case(test_case, self)
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
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
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Statements:\n{prettify_list(self.statements)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

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
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(
                    prompt=prompt, schema=Reason
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

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

        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt=prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdicts(
        self, input: str
    ) -> List[AnswerRelvancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                return [item for item in res.verdicts]
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return [
                    AnswerRelvancyVerdict(**item) for item in data["verdicts"]
                ]

    def _generate_verdicts(self, input: str) -> List[AnswerRelvancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = AnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                return [item for item in res.verdicts]
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return [
                    AnswerRelvancyVerdict(**item) for item in data["verdicts"]
                ]

    async def _a_generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_statements(
            actual_output=actual_output,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["statements"]
        else:
            try:
                res: Statements = await self.model.a_generate(
                    prompt, schema=Statements
                )
                return res.statements
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["statements"]

    def _generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = AnswerRelevancyTemplate.generate_statements(
            actual_output=actual_output,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["statements"]
        else:
            try:
                res: Statements = self.model.generate(prompt, schema=Statements)
                return res.statements
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["statements"]

    def _calculate_score(self):
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
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
