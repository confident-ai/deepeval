from typing import Optional, List, Union, Dict

from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.step_efficiency.template import (
    StepEfficiencyTemplate,
)
from deepeval.metrics.step_efficiency.schema import Task, EfficiencyVerdict
from deepeval.metrics.api import metric_data_manager


class StepEfficiencyMetric(BaseMetric):

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
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
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.requires_trace = True

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ):
        has_trace: bool = isinstance(test_case._trace_dict, Dict)
        if not has_trace:
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
                task = self._extract_task_from_trace(test_case)
                efficiency_verdict = self._get_score(task, test_case)
                self.score = (
                    0
                    if self.strict_mode
                    and efficiency_verdict.score < self.threshold
                    else efficiency_verdict.score
                )
                self.reason = efficiency_verdict.reason
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Task: {task} \n",
                        f"Efficiency Score: {self.score}",
                        f"Efficiency Reason: {self.reason}",
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
    ):
        has_trace: bool = isinstance(test_case._trace_dict, Dict)
        if not has_trace:
            check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            task = await self._a_extract_task_from_trace(test_case)
            efficiency_verdict = await self._a_get_score(task, test_case)
            self.score = (
                0
                if self.strict_mode
                and efficiency_verdict.score < self.threshold
                else efficiency_verdict.score
            )
            self.reason = efficiency_verdict.reason
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Task: {task} \n",
                    f"Efficiency Score: {self.score}",
                    f"Efficiency Reason: {self.reason}",
                ],
            )

            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

            return self.score

    def _get_score(self, task: str, test_case: LLMTestCase):
        if test_case._trace_dict is not None:
            prompt = StepEfficiencyTemplate.get_execution_efficiency(
                task, test_case._trace_dict
            )

        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=EfficiencyVerdict)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: Task = self.model.generate(
                    prompt, schema=EfficiencyVerdict
                )
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return EfficiencyVerdict(**data)

    async def _a_get_score(self, task: str, test_case: LLMTestCase):
        if test_case._trace_dict is not None:
            prompt = StepEfficiencyTemplate.get_execution_efficiency(
                task, test_case._trace_dict
            )

        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=EfficiencyVerdict
            )
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: Task = await self.model.a_generate(
                    prompt, schema=EfficiencyVerdict
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return EfficiencyVerdict(**data)

    def _extract_task_from_trace(self, test_case: LLMTestCase) -> str:
        prompt = StepEfficiencyTemplate.extract_task_from_trace(
            test_case._trace_dict
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Task)
            self.evaluation_cost += cost
            return res.task
        else:
            try:
                res: Task = self.model.generate(prompt, schema=Task)
                return res.task
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["task"]

    async def _a_extract_task_from_trace(self, test_case: LLMTestCase) -> str:
        prompt = StepEfficiencyTemplate.extract_task_from_trace(
            test_case._trace_dict
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Task)
            self.evaluation_cost += cost
            return res.task
        else:
            try:
                res: Task = await self.model.a_generate(prompt, schema=Task)
                return res.task
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["task"]

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
        return "Execution Efficiency"
