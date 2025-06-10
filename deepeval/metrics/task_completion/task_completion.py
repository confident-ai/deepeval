from typing import Optional, List, Tuple, Union

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
    ConversationalTestCase,
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
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
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
                    )
                )
            else:
                user_goal, task_outcome = self._extract_goal_and_outcome(
                    test_case
                )
                self.user_goal = user_goal
                self.task_outcome = task_outcome
                verdict, reason = self._generate_verdicts()
                self.verdict = verdict
                self.reason = reason
                self.score = self._calculate_score()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"User Goal: {user_goal}",
                        f"Task Outcome: {task_outcome}",
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
            user_goal, task_outcome = await self._a_extract_goal_and_outcome(
                test_case
            )
            self.user_goal = user_goal
            self.task_outcome = task_outcome
            verdict, reason = await self._a_generate_verdicts()
            self.verdict = verdict
            self.reason = reason
            self.score = self._calculate_score()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"User Goal: {user_goal}",
                    f"Task Outcome: {task_outcome}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_verdicts(self) -> Tuple:
        prompt = TaskCompletionTemplate.generate_verdict(
            user_goal=self.user_goal,
            actual_outcome=self.task_outcome,
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
            user_goal=self.user_goal,
            actual_outcome=self.task_outcome,
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

    async def _a_extract_goal_and_outcome(
        self,
        test_case: LLMTestCase,
    ) -> Tuple:
        prompt = TaskCompletionTemplate.extract_goal_and_outcome(
            input=test_case.input,
            actual_output=test_case.actual_output,
            tools_called=test_case.tools_called,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=GoalAndOutcome
            )
            self.evaluation_cost += cost
            return res.user_goal, res.task_outcome
        else:
            try:
                res: GoalAndOutcome = await self.model.a_generate(
                    prompt, schema=GoalAndOutcome
                )
                return res.user_goal, res.task_outcome
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["user_goal"], data["task_outcome"]

    def _extract_goal_and_outcome(
        self,
        test_case: LLMTestCase,
    ) -> Tuple:
        prompt = TaskCompletionTemplate.extract_goal_and_outcome(
            input=test_case.input,
            actual_output=test_case.actual_output,
            tools_called=test_case.tools_called,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=GoalAndOutcome)
            self.evaluation_cost += cost
            return res.user_goal, res.task_outcome
        else:
            try:
                res: GoalAndOutcome = self.model.generate(
                    prompt, schema=GoalAndOutcome
                )
                return res.user_goal, res.task_outcome
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["user_goal"], data["task_outcome"]

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
