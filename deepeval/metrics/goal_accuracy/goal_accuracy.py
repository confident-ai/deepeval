from typing import Optional, List, Union
import asyncio
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    get_unit_interactions,
    print_tools_called,
    check_conversational_test_case_params,
    initialize_model,
)
from deepeval.test_case import ConversationalTestCase, TurnParams, Turn
from deepeval.metrics import BaseConversationalMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.goal_accuracy.template import (
    GoalAccuracyTemplate,
)
from deepeval.metrics.goal_accuracy.schema import (
    GoalSteps,
    GoalScore,
    PlanScore,
)
from deepeval.metrics.api import metric_data_manager


class GoalAccuracyMetric(BaseConversationalMetric):

    _required_test_case_params = [
        TurnParams.ROLE,
        TurnParams.CONTENT,
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
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ):
        check_conversational_test_case_params(
            test_case, self._required_test_case_params, self
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
                unit_interactions = get_unit_interactions(test_case.turns)
                goal_and_steps_taken = self._goal_and_steps_taken(
                    unit_interactions
                )
                goal_scores = [
                    self._get_goal_accuracy_score(
                        task.user_goal, task.steps_taken
                    )
                    for task in goal_and_steps_taken
                ]
                plan_scores = [
                    self._get_plan_scores(task.user_goal, task.steps_taken)
                    for task in goal_and_steps_taken
                ]
                self.score = self._calculate_score(goal_scores, plan_scores)
                self.success = self.score >= self.threshold
                self.reason = self._generate_reason(goal_scores, plan_scores)

                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Goals and steps taken: \n{self.print_goals_and_steps_taken(goal_and_steps_taken)} \n",
                        f"Goal evaluations: {prettify_list(goal_scores)} \n\n"
                        f"Plan evaluations: {prettify_list(plan_scores)} \n\n"
                        f"Final Score: {self.score}",
                        f"Final Reason: {self.reason}",
                    ],
                )

                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )

                return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ):
        check_conversational_test_case_params(
            test_case, self._required_test_case_params, self
        )

        self.evaluation_cost = 0 if self.using_native_model else None

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            unit_interactions = get_unit_interactions(test_case.turns)
            goal_and_steps_taken = self._goal_and_steps_taken(unit_interactions)
            goal_scores = await asyncio.gather(
                *[
                    self._a_get_goal_accuracy_score(
                        task.user_goal, task.steps_taken
                    )
                    for task in goal_and_steps_taken
                ]
            )
            plan_scores = await asyncio.gather(
                *[
                    self._a_get_plan_scores(task.user_goal, task.steps_taken)
                    for task in goal_and_steps_taken
                ]
            )
            self.score = self._calculate_score(goal_scores, plan_scores)
            self.success = self.score >= self.threshold
            self.reason = await self._a_generate_reason(
                goal_scores, plan_scores
            )

            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Goals and steps taken: \n{self.print_goals_and_steps_taken(goal_and_steps_taken)} \n",
                    f"Goal evaluations: {prettify_list(goal_scores)} \n\n"
                    f"Plan evaluations: {prettify_list(plan_scores)} \n\n"
                    f"Final Score: {self.score}",
                    f"Final Reason: {self.reason}",
                ],
            )

            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

            return self.score

    def _goal_and_steps_taken(
        self, unit_interactions: List[List[Turn]]
    ) -> List[GoalSteps]:
        goal_and_steps_taken = []
        for unit_interaction in unit_interactions:
            user_messages = "User messages: \n"
            for turn in unit_interaction:
                if turn.role == "user":
                    user_messages += turn.content + "\n"
                else:
                    break
            new_goal_steps = GoalSteps(user_goal=user_messages, steps_taken=[])
            assistant_messages = "Assistant messages: \n"
            for turn in unit_interaction[1:]:
                if turn.role == "assistant":
                    assistant_messages += f"{turn.content} \n"
                    if turn.tools_called:
                        assistant_messages += f"Tools called: \n{print_tools_called(turn.tools_called)} \n"
                    new_goal_steps.steps_taken.append(assistant_messages)
            goal_and_steps_taken.append(new_goal_steps)
        return goal_and_steps_taken

    def _get_plan_scores(self, user_goal, steps_taken):
        prompt = GoalAccuracyTemplate.get_plan_evaluation_score(
            user_goal, "\n".join(steps_taken)
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=PlanScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: PlanScore = self.model.generate(prompt, schema=PlanScore)
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return PlanScore(**data)

    async def _a_get_plan_scores(self, user_goal, steps_taken):
        prompt = GoalAccuracyTemplate.get_plan_evaluation_score(
            user_goal, "\n".join(steps_taken)
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=PlanScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: PlanScore = await self.model.a_generate(
                    prompt, schema=PlanScore
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return PlanScore(**data)

    def _calculate_score(
        self, goal_scores: List[GoalScore], plan_scores: List[PlanScore]
    ):
        goal_scores = [goal_score.score for goal_score in goal_scores]
        plan_scores = [plan_score.score for plan_score in plan_scores]
        goal_score_divisor = len(goal_scores) if len(goal_scores) > 0 else 1
        plan_score_divisor = len(plan_scores) if len(plan_scores) > 0 else 1
        goal_avg = sum(goal_scores) / goal_score_divisor
        plan_avg = sum(plan_scores) / plan_score_divisor
        score = (goal_avg + plan_avg) / 2
        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_reason(
        self, goal_scores: List[GoalScore], plan_scores: List[PlanScore]
    ):
        goal_evaluations = ""
        for goal_score in goal_scores:
            goal_evaluations += (
                f"Score: {goal_score.score}, Reason: {goal_score.reason}"
            )
        plan_evalautions = ""
        for plan_score in plan_scores:
            plan_evalautions += (
                f"Score: {plan_score.score}, Reason: {plan_score.reason} \n"
            )

        prompt = GoalAccuracyTemplate.get_final_reason(
            self.score, self.threshold, goal_evaluations, plan_evalautions
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            return res
        else:
            res = self.model.generate(prompt)
            return res

    async def _a_generate_reason(
        self, goal_scores: List[GoalScore], plan_scores: List[PlanScore]
    ):
        goal_evaluations = ""
        for goal_score in goal_scores:
            goal_evaluations += (
                f"Score: {goal_score.score}, Reason: {goal_score.reason}"
            )
        plan_evalautions = ""
        for plan_score in plan_scores:
            plan_evalautions += (
                f"Score: {plan_score.score}, Reason: {plan_score.reason} \n"
            )

        prompt = GoalAccuracyTemplate.get_final_reason(
            self.score, self.threshold, goal_evaluations, plan_evalautions
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            return res
        else:
            res = await self.model.a_generate(prompt)
            return res

    def _get_goal_accuracy_score(self, user_goal, steps_taken):
        prompt = GoalAccuracyTemplate.get_accuracy_score(
            user_goal, "\n".join(steps_taken)
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=GoalScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: GoalScore = self.model.generate(prompt, schema=GoalScore)
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return GoalScore(**data)

    async def _a_get_goal_accuracy_score(self, user_goal, steps_taken):
        prompt = GoalAccuracyTemplate.get_accuracy_score(
            user_goal, "\n".join(steps_taken)
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=GoalScore)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res: GoalScore = await self.model.a_generate(
                    prompt, schema=GoalScore
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return GoalScore(**data)

    def print_goals_and_steps_taken(self, goals_and_steps):
        final_goals_and_steps = ""
        for goal_step in goals_and_steps:
            final_goals_and_steps += f"{goal_step.user_goal} \n"
            final_goals_and_steps += (
                f"c{prettify_list(goal_step.steps_taken)} \n\n"
            )
        return final_goals_and_steps

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
        return "Goal Accuracy"
