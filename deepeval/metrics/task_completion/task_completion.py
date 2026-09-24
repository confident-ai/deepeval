from typing import Optional, List, Tuple, Union, Dict, Type

from deepeval.utils import get_or_create_event_loop, serialize_to_json
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    initialize_system_one_model,
    print_tools_called,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    SystemOneEvalSpec,
    SystemOneScoreSpec,
    compact_trace,
    format_decision_reason,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    system_one_score,
    a_system_one_score,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.task_completion.schema import (
    TaskAndOutcome,
    TaskCompletionVerdict,
)
from deepeval.templates import make_template_class


TaskCompletionTemplate = make_template_class("TaskCompletionMetric")

TASK_COMPLETION_LEVELS = [
    "Not achieved",
    "Partly achieved",
    "Mostly achieved",
    "Fully achieved",
]


class TaskCompletionMetric(BaseMetric):

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        task: Optional[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[EvalModeName] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[
            TaskCompletionTemplate
        ] = TaskCompletionTemplate,
    ):
        if task is None:
            self._is_task_provided = False
        else:
            self._is_task_provided = True

        self.task = task
        self.threshold = 1 if strict_mode else threshold
        self.eval_mode = resolve_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.requires_trace = True
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
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
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
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
                if run_system_one_eval(self, test_case):
                    return self.score

                task, self.outcome = self._extract_task_and_outcome(test_case)
                if self.task is None or not self._is_task_provided:
                    self.task = task
                self.verdict, self.reason = self._generate_verdicts()
                self.score = self._calculate_score()
                self.success = self.is_successful()
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
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self.score

            task, self.outcome = await self._a_extract_task_and_outcome(
                test_case
            )
            if self.task is None or not self._is_task_provided:
                self.task = task
            self.verdict, self.reason = await self._a_generate_verdicts()
            self.score = self._calculate_score()
            self.success = self.is_successful()
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
        value = await a_system_one_score(self, self._system_one_score_spec())
        if value is not None:
            return value, self._system_one_reason(value)
        prompt = self._get_prompt(
            "generate_verdict",
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
        value = system_one_score(self, self._system_one_score_spec())
        if value is not None:
            return value, self._system_one_reason(value)
        prompt = self._get_prompt(
            "generate_verdict",
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
            prompt = self._get_prompt(
                "extract_task_and_outcome_from_trace",
                trace_json=serialize_to_json(test_case._trace_dict, indent=2),
            )
        else:
            # TODO: Deprecate this soon
            prompt = self._get_prompt(
                "extract_goal_and_outcome",
                input=test_case.input,
                actual_output=test_case.actual_output,
                tools_called_formatted=print_tools_called(
                    test_case.tools_called
                ),
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
            prompt = self._get_prompt(
                "extract_task_and_outcome_from_trace",
                trace_json=serialize_to_json(test_case._trace_dict, indent=2),
            )
        else:
            # TODO: Deprecate this soon
            prompt = self._get_prompt(
                "extract_goal_and_outcome",
                input=test_case.input,
                actual_output=test_case.actual_output,
                tools_called_formatted=print_tools_called(
                    test_case.tools_called
                ),
            )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=TaskAndOutcome,
            extract_schema=lambda s: (s.task, s.outcome),
            extract_json=lambda data: (data["task"], data["outcome"]),
        )

    def _system_one_score_spec(self) -> SystemOneScoreSpec:
        return SystemOneScoreSpec(
            instructions=self._get_prompt("_experimental_system_one_score"),
            levels=TASK_COMPLETION_LEVELS,
            state={"task": self.task, "outcome": self.outcome},
        )

    def _system_one_reason(self, value: float) -> Optional[str]:
        if not self.include_reason:
            return None
        return format_decision_reason(self, "task completion", value)

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        the trace (or `input`, `actual_output` and `tools_called` when there
        is none); see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        has_trace = isinstance(test_case._trace_dict, Dict)
        task = self.task if self._is_task_provided else None
        return SystemOneEvalSpec(
            evaluation_params=(
                []
                if has_trace
                else [
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.TOOLS_CALLED,
                ]
            ),
            questions=parse_questions(
                self._get_prompt(
                    "_experimental_system_one_questions",
                    has_trace=has_trace,
                    has_task=task is not None,
                )
            ),
            extra_state={
                "trace": (
                    compact_trace(test_case._trace_dict) if has_trace else None
                ),
                "task": task,
            },
        )

    def _calculate_score(self):
        return (
            0
            if self.strict_mode and self.verdict < self.threshold
            else self.verdict
        )

    @property
    def __name__(self):
        return "Task Completion"
