from typing import List, Optional, Type, Union

from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.metrics.non_advice.template import NonAdviceTemplate
from deepeval.metrics.non_advice.schema import *
from deepeval.metrics.api import metric_data_manager


class NonAdviceMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        advice_types: List[str],  # Required parameter - no defaults
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[NonAdviceTemplate] = NonAdviceTemplate,
    ):
        if not advice_types or len(advice_types) == 0:
            raise ValueError(
                "advice_types must be specified and non-empty. "
                "Examples: ['financial'], ['medical'], ['legal'], "
                "or ['financial', 'medical'] for multiple types."
            )

        self.threshold = 1 if strict_mode else threshold
        self.advice_types = advice_types
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

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
                        _log_metric_to_confident=_log_metric_to_confident,
                    )
                )
            else:
                self.advices: List[str] = self._generate_advices(
                    test_case.actual_output
                )
                self.verdicts: List[NonAdviceVerdict] = (
                    self._generate_verdicts()
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Advices:\n{prettify_list(self.advices)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.advices: List[str] = await self._a_generate_advices(
                test_case.actual_output
            )
            self.verdicts: List[NonAdviceVerdict] = (
                await self._a_generate_verdicts()
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Advices:\n{prettify_list(self.advices)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        non_advice_violations = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                non_advice_violations.append(verdict.reason)

        prompt: dict = self.evaluation_template.generate_reason(
            non_advice_violations=non_advice_violations,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=NonAdviceScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: NonAdviceScoreReason = await self.model.a_generate(
                    prompt, schema=NonAdviceScoreReason
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        non_advice_violations = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                non_advice_violations.append(verdict.reason)

        prompt: dict = self.evaluation_template.generate_reason(
            non_advice_violations=non_advice_violations,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=NonAdviceScoreReason)
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: NonAdviceScoreReason = self.model.generate(
                    prompt, schema=NonAdviceScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdicts(self) -> List[NonAdviceVerdict]:
        if len(self.advices) == 0:
            return []

        verdicts: List[NonAdviceVerdict] = []
        prompt = self.evaluation_template.generate_verdicts(
            advices=self.advices
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    NonAdviceVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(self) -> List[NonAdviceVerdict]:
        if len(self.advices) == 0:
            return []

        verdicts: List[NonAdviceVerdict] = []
        prompt = self.evaluation_template.generate_verdicts(
            advices=self.advices
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    NonAdviceVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    async def _a_generate_advices(self, actual_output: str) -> List[str]:
        prompt = self.evaluation_template.generate_advices(
            actual_output=actual_output, advice_types=self.advice_types
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Advices)
            self.evaluation_cost += cost
            return res.advices
        else:
            try:
                res: Advices = await self.model.a_generate(
                    prompt, schema=Advices
                )
                return res.advices
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["advices"]

    def _generate_advices(self, actual_output: str) -> List[str]:
        prompt = self.evaluation_template.generate_advices(
            actual_output=actual_output, advice_types=self.advice_types
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Advices)
            self.evaluation_cost += cost
            return res.advices
        else:
            try:
                res: Advices = self.model.generate(prompt, schema=Advices)
                return res.advices
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["advices"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        appropriate_advice_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                appropriate_advice_count += 1

        score = appropriate_advice_count / number_of_verdicts
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
        return "Non-Advice"
