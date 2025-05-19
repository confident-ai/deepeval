from typing import Optional, List, Union

from deepeval.metrics import BaseMultimodalMetric
from deepeval.test_case import MLLMTestCaseParams, MLLMTestCase, MLLMImage
from deepeval.metrics.multimodal_metrics.multimodal_answer_relevancy.template import (
    MultimodalAnswerRelevancyTemplate,
)
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_mllm_test_case_params,
    initialize_multimodal_model,
)
from deepeval.models import DeepEvalBaseMLLM
from deepeval.metrics.multimodal_metrics.multimodal_answer_relevancy.schema import *
from deepeval.metrics.indicator import metric_progress_indicator


class MultimodalAnswerRelevancyMetric(BaseMultimodalMetric):

    _required_params: List[MLLMTestCaseParams] = [
        MLLMTestCaseParams.INPUT,
        MLLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseMLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_multimodal_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_mllm_test_case_params(
            test_case, self._required_params, None, None, self
        )
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                self.statements: List[str] = self._generate_statements(
                    test_case.actual_output
                )
                self.verdicts: List[AnswerRelevancyVerdict] = (
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
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_mllm_test_case_params(
            test_case, self._required_params, None, None, self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.statements: List[str] = await self._a_generate_statements(
                test_case.actual_output
            )
            self.verdicts: List[AnswerRelevancyVerdict] = (
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

    async def _a_generate_reason(
        self,
        input: List[Union[str, MLLMImage]],
    ) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = MultimodalAnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Reason)
            self.evaluation_cost += cost
            return res.reason
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

    def _generate_reason(
        self,
        input: List[Union[str, MLLMImage]],
    ) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = MultimodalAnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Reason)
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: Reason = self.model.generate(prompt=prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdicts(
        self,
        input: List[Union[str, MLLMImage]],
    ) -> List[AnswerRelevancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = MultimodalAnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            return [item for item in res.verdicts]
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
                    AnswerRelevancyVerdict(**item) for item in data["verdicts"]
                ]

    def _generate_verdicts(
        self, input: List[Union[str, MLLMImage]]
    ) -> List[AnswerRelevancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = MultimodalAnswerRelevancyTemplate.generate_verdicts(
            input=input,
            actual_output=self.statements,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            return [item for item in res.verdicts]
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                return [item for item in res.verdicts]
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return [
                    AnswerRelevancyVerdict(**item) for item in data["verdicts"]
                ]

    async def _a_generate_statements(
        self,
        actual_output: List[Union[str, MLLMImage]],
    ) -> List[str]:
        prompt = MultimodalAnswerRelevancyTemplate.generate_statements(
            actual_output=[
                ele for ele in actual_output if isinstance(ele, str)
            ],
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Statements)
            self.evaluation_cost += cost
            statements: List[str] = res.statements + [
                ele for ele in actual_output if isinstance(ele, MLLMImage)
            ]
            return statements
        else:
            try:
                res: Statements = await self.model.a_generate(
                    prompt, schema=Statements
                )
                statements: List[str] = res.statements + [
                    ele for ele in actual_output if isinstance(ele, MLLMImage)
                ]
                return statements
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                statements = data["statements"] + [
                    ele for ele in actual_output if isinstance(ele, MLLMImage)
                ]
                return statements

    def _generate_statements(
        self,
        actual_output: List[Union[str, MLLMImage]],
    ) -> List[str]:
        prompt = MultimodalAnswerRelevancyTemplate.generate_statements(
            actual_output=[
                ele for ele in actual_output if isinstance(ele, str)
            ],
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Statements)
            self.evaluation_cost += cost
            statements = res.statements + [
                ele for ele in actual_output if isinstance(ele, MLLMImage)
            ]
            return statements
        else:
            try:
                res: Statements = self.model.generate(prompt, schema=Statements)
                statements = res.statements + [
                    ele for ele in actual_output if isinstance(ele, MLLMImage)
                ]
                return statements
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                statements = data["statements"] + [
                    ele for ele in actual_output if isinstance(ele, MLLMImage)
                ]
                return statements

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
        return "Multimodal Answer Relevancy"
