from typing import Optional, List, Tuple, Union, Dict

from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.task_completion.template import TaskCompletionTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.task_completion.schema import (
    TaskAndOutcome,
    TaskCompletionVerdict,
)
from deepeval.metrics.api import metric_data_manager


class TaskCompletionMetric(BaseMetric):

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
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
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
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
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )

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

            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

            return self.score

    async def _a_generate_verdicts(self) -> Tuple:
        prompt = TaskCompletionTemplate.generate_verdict(
            task=self.task,
            actual_outcome=self.outcome,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=TaskCompletionVerdict,
            extract_schema=lambda s: (s.verdict, s.reason),
            extract_json=lambda data: (data["verdict"], data["reason"]),
        )

    def _generate_verdicts(self) -> Tuple:
        prompt = TaskCompletionTemplate.generate_verdict(
            task=self.task,
            actual_outcome=self.outcome,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=TaskCompletionVerdict,
            extract_schema=lambda s: (s.verdict, s.reason),
            extract_json=lambda data: (data["verdict"], data["reason"]),
        )

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
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=TaskAndOutcome,
            extract_schema=lambda s: (s.task, s.outcome),
            extract_json=lambda data: (data["task"], data["outcome"]),
        )

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
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=TaskAndOutcome,
            extract_schema=lambda s: (s.task, s.outcome),
            extract_json=lambda data: (data["task"], data["outcome"]),
        )

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
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Task Completion"
