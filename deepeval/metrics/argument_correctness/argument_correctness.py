from typing import Optional, List, Type, Union

from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ToolCall,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.argument_correctness.template import (
    ArgumentCorrectnessTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.argument_correctness.schema import *


class ArgumentCorrectnessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.TOOLS_CALLED,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[
            ArgumentCorrectnessTemplate
        ] = ArgumentCorrectnessTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
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
    ) -> float:

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
                check_llm_test_case_params(test_case, self._required_params, self)

                if len(test_case.tools_called) == 0:
                    self.verdicts = []
                    self.score = 1.0
                    self.reason = "No tool calls provided"
                else:
                    self.verdicts: List[ArgumentCorrectnessVerdict] = (
                        self._generate_verdicts(
                            test_case.input, test_case.tools_called
                        )
                    )
                    self.score = self._calculate_score()
                    self.reason = self._generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if len(test_case.tools_called) == 0:
                self.verdicts = []
                self.score = 1.0
                self.reason = "No tool calls provided"
            else:
                self.verdicts: List[ArgumentCorrectnessVerdict] = (
                    await self._a_generate_verdicts(
                        test_case.input, test_case.tools_called
                    )
                )
                self.score = self._calculate_score()
                self.reason = await self._a_generate_reason(test_case.input)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def _a_generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        incorrect_tool_calls_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                incorrect_tool_calls_reasons.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            incorrect_tool_calls_reasons=incorrect_tool_calls_reasons,
            input=input,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=ArgumentCorrectnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ArgumentCorrectnessScoreReason = (
                    await self.model.a_generate(
                        prompt=prompt, schema=ArgumentCorrectnessScoreReason
                    )
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        incorrect_tool_calls_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                incorrect_tool_calls_reasons.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            incorrect_tool_calls_reasons=incorrect_tool_calls_reasons,
            input=input,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=ArgumentCorrectnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ArgumentCorrectnessScoreReason = self.model.generate(
                    prompt=prompt, schema=ArgumentCorrectnessScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdicts(
        self,
        input: str,
        tools_called: List[ToolCall],
    ) -> List[ArgumentCorrectnessVerdict]:
        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            tools_called=tools_called,
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
                    ArgumentCorrectnessVerdict(**item)
                    for item in data["verdicts"]
                ]

    def _generate_verdicts(
        self,
        input: str,
        tools_called: List[ToolCall],
    ) -> List[ArgumentCorrectnessVerdict]:
        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            tools_called=tools_called,
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
                    ArgumentCorrectnessVerdict(**item)
                    for item in data["verdicts"]
                ]

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        correct_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                correct_count += 1

        score = correct_count / number_of_verdicts
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
        return "Argument Correctness"
