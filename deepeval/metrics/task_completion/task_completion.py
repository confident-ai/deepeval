from typing import Optional, List, Tuple, Union, Dict

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
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.task_completion.template import TaskCompletionTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.task_completion.schema import *


class TaskCompletionMetric(BaseMetric):

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.TOOLS_CALLED,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        task: Optional[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        if task is None:
            self._is_task_provided = False
        else:
            self._is_task_provided = True

        self.task = task
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
    ) -> float:
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
                task, self.outcome = self._extract_task_and_outcome(test_case)
                if self.task is None or not self._is_task_provided:
                    self.task = task
                self.verdict, self.reason = self._generate_verdicts()
                self.score = self._calculate_score()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Task: {self.task}",
                        f"Outcome: {self.outcome}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
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
            task, self.outcome = await self._a_extract_task_and_outcome(
                test_case
            )
            if self.task is None or not self._is_task_provided:
                self.task = task
            self.verdict, self.reason = await self._a_generate_verdicts()
            self.score = self._calculate_score()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Task: {self.task}",
                    f"Outcome: {self.outcome}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_verdicts(self) -> Tuple:
        prompt = TaskCompletionTemplate.generate_verdict(
            task=self.task,
            actual_outcome=self.outcome,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=TaskCompletionVerdict
            )
            self.evaluation_cost += cost
            return res.verdict, res.reason
        else:
            try:
                res: TaskCompletionVerdict = await self.model.a_generate(
                    prompt, schema=TaskCompletionVerdict
                )
                return res.verdict, res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["verdict"], data["reason"]

    def _generate_verdicts(self) -> Tuple:
        prompt = TaskCompletionTemplate.generate_verdict(
            task=self.task,
            actual_outcome=self.outcome,
        )
        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=TaskCompletionVerdict
            )
            self.evaluation_cost += cost
            return res.verdict, res.reason
        else:
            try:
                res: TaskCompletionVerdict = self.model.generate(
                    prompt, schema=TaskCompletionVerdict
                )
                return res.verdict, res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["verdict"], data["reason"]

    async def _a_extract_task_and_outcome(
        self,
        test_case: LLMTestCase,
    ) -> Tuple:
        has_trace: bool = isinstance(test_case._trace_dict, Dict)
        if has_trace:
            prompt = TaskCompletionTemplate.extract_task_and_outcome_from_trace(
                trace=test_case._trace_dict
            )
        else:
            # TODO: Deprecate this soon
            prompt = TaskCompletionTemplate.extract_goal_and_outcome(
                input=test_case.input,
                actual_output=test_case.actual_output,
                tools_called=test_case.tools_called,
            )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=TaskAndOutcome
            )
            self.evaluation_cost += cost
            return res.task, res.outcome
        else:
            try:
                res: TaskAndOutcome = await self.model.a_generate(
                    prompt, schema=TaskAndOutcome
                )
                return res.task, res.outcome
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["task"], data["outcome"]

    def _extract_task_and_outcome(
        self,
        test_case: LLMTestCase,
    ) -> Tuple:
        has_trace: bool = isinstance(test_case._trace_dict, Dict)
        if has_trace:
            prompt = TaskCompletionTemplate.extract_task_and_outcome_from_trace(
                trace=test_case._trace_dict
            )
        else:
            # TODO: Deprecate this soon
            prompt = TaskCompletionTemplate.extract_goal_and_outcome(
                input=test_case.input,
                actual_output=test_case.actual_output,
                tools_called=test_case.tools_called,
            )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=TaskAndOutcome)
            self.evaluation_cost += cost
            return res.task, res.outcome
        else:
            try:
                res: TaskAndOutcome = self.model.generate(
                    prompt, schema=TaskAndOutcome
                )
                return res.task, res.outcome
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["task"], data["outcome"]

    def _calculate_score(self):
        return (
            0
            if self.strict_mode and self.verdict < self.threshold
            else self.verdict
        )

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
        return "Task Completion"
