from typing import Optional, List, Union, Type
import asyncio
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    get_unit_interactions,
    print_tools_called,
    check_conversational_test_case_params,
    initialize_model,
    initialize_system_one_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    accrue_token_usage,
    SystemOneEvalSpec,
    SystemOneScoreSpec,
    format_decision_reason,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    system_one_score,
    a_system_one_score,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn
from deepeval.metrics import BaseConversationalMetric
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.goal_accuracy.schema import (
    GoalSteps,
    GoalScore,
    PlanScore,
)
from deepeval.templates import make_template_class


GoalAccuracyTemplate = make_template_class("GoalAccuracyMetric")

GOAL_ACCURACY_LEVELS = [
    "Not achieved",
    "Weak attempt",
    "Partially achieved",
    "Mostly achieved",
    "Fully achieved",
]

PLAN_QUALITY_LEVELS = [
    "No plan",
    "Weak or fragmented plan, rarely followed",
    "Partial plan, partly followed",
    "Clear plan, mostly followed",
    "Complete plan, fully followed",
]


class GoalAccuracyMetric(BaseConversationalMetric):

    _required_test_case_params = [
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
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
        evaluation_template: Type[GoalAccuracyTemplate] = GoalAccuracyTemplate,
    ):
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
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ):
        multimodal = test_case.multimodal
        check_conversational_test_case_params(
            test_case,
            self._required_test_case_params,
            self,
            None,
            self.model,
            multimodal,
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

                unit_interactions = get_unit_interactions(test_case.turns)
                goal_and_steps_taken = self._goal_and_steps_taken(
                    unit_interactions
                )
                goal_scores = [
                    self._get_goal_accuracy_score(
                        task.user_goal, task.steps_taken, multimodal
                    )
                    for task in goal_and_steps_taken
                ]
                plan_scores = [
                    self._get_plan_scores(
                        task.user_goal, task.steps_taken, multimodal
                    )
                    for task in goal_and_steps_taken
                ]
                self.score = self._calculate_score(goal_scores, plan_scores)
                self.success = self.is_successful()
                self.reason = self._generate_reason(
                    goal_scores, plan_scores, multimodal
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

                return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ):
        multimodal = test_case.multimodal
        check_conversational_test_case_params(
            test_case,
            self._required_test_case_params,
            self,
            None,
            self.model,
            multimodal,
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

            unit_interactions = get_unit_interactions(test_case.turns)
            goal_and_steps_taken = self._goal_and_steps_taken(unit_interactions)
            goal_scores = await asyncio.gather(
                *[
                    self._a_get_goal_accuracy_score(
                        task.user_goal, task.steps_taken, multimodal
                    )
                    for task in goal_and_steps_taken
                ]
            )
            plan_scores = await asyncio.gather(
                *[
                    self._a_get_plan_scores(
                        task.user_goal, task.steps_taken, multimodal
                    )
                    for task in goal_and_steps_taken
                ]
            )
            self.score = self._calculate_score(goal_scores, plan_scores)
            self.success = self.is_successful()
            self.reason = await self._a_generate_reason(
                goal_scores, plan_scores, multimodal
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

    def _get_plan_scores(self, user_goal, steps_taken, multimodal: bool):
        value = system_one_score(
            self,
            self._experimental_system_one_spec(
                "plan", user_goal, steps_taken, multimodal
            ),
        )
        if value is not None:
            return PlanScore(
                score=value, reason=self._system_one_reason("plan", value)
            )
        prompt = self._get_prompt(
            "get_plan_evaluation_score",
            task=user_goal,
            steps_taken="\n".join(steps_taken),
            multimodal=multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=PlanScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: PlanScore(**data),
        )

    async def _a_get_plan_scores(
        self, user_goal, steps_taken, multimodal: bool
    ):
        value = await a_system_one_score(
            self,
            self._experimental_system_one_spec(
                "plan", user_goal, steps_taken, multimodal
            ),
        )
        if value is not None:
            return PlanScore(
                score=value, reason=self._system_one_reason("plan", value)
            )
        prompt = self._get_prompt(
            "get_plan_evaluation_score",
            task=user_goal,
            steps_taken="\n".join(steps_taken),
            multimodal=multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=PlanScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: PlanScore(**data),
        )

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
        self,
        goal_scores: List[GoalScore],
        plan_scores: List[PlanScore],
        multimodal: bool,
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

        prompt = self._get_prompt(
            "get_final_reason",
            final_score=self.score,
            threshold=self.threshold,
            goal_evaluations=goal_evaluations,
            plan_evalautions=plan_evalautions,
            multimodal=multimodal,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self._accrue_cost(cost)
            accrue_token_usage(self, cost)
            return res
        else:
            res = self.model.generate(prompt)
            return res

    async def _a_generate_reason(
        self,
        goal_scores: List[GoalScore],
        plan_scores: List[PlanScore],
        multimodal: bool,
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

        prompt = self._get_prompt(
            "get_final_reason",
            final_score=self.score,
            threshold=self.threshold,
            goal_evaluations=goal_evaluations,
            plan_evalautions=plan_evalautions,
            multimodal=multimodal,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self._accrue_cost(cost)
            accrue_token_usage(self, cost)
            return res
        else:
            res = await self.model.a_generate(prompt)
            return res

    def _get_goal_accuracy_score(
        self, user_goal, steps_taken, multimodal: bool
    ):
        value = system_one_score(
            self,
            self._experimental_system_one_spec(
                "goal", user_goal, steps_taken, multimodal
            ),
        )
        if value is not None:
            return GoalScore(
                score=value, reason=self._system_one_reason("goal", value)
            )
        prompt = self._get_prompt(
            "get_accuracy_score",
            task=user_goal,
            steps_taken="\n".join(steps_taken),
            multimodal=multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=GoalScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: GoalScore(**data),
        )

    async def _a_get_goal_accuracy_score(
        self, user_goal, steps_taken, multimodal: bool
    ):
        value = await a_system_one_score(
            self,
            self._experimental_system_one_spec(
                "goal", user_goal, steps_taken, multimodal
            ),
        )
        if value is not None:
            return GoalScore(
                score=value, reason=self._system_one_reason("goal", value)
            )
        prompt = self._get_prompt(
            "get_accuracy_score",
            task=user_goal,
            steps_taken="\n".join(steps_taken),
            multimodal=multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=GoalScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: GoalScore(**data),
        )

    def _experimental_system_one_spec(
        self, kind: str, user_goal, steps_taken, multimodal: bool
    ) -> Optional[SystemOneScoreSpec]:
        if multimodal:
            return None
        return SystemOneScoreSpec(
            instructions=self._get_prompt(
                f"_experimental_system_one_{kind}_score"
            ),
            levels=(
                GOAL_ACCURACY_LEVELS if kind == "goal" else PLAN_QUALITY_LEVELS
            ),
            state={"task": user_goal, "steps_taken": list(steps_taken)},
        )

    def _system_one_reason(self, kind: str, value: float) -> str:
        what = "goal accuracy" if kind == "goal" else "plan quality"
        return format_decision_reason(self, what, value)

    def _system_one_eval_spec(
        self, test_case: ConversationalTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole conversation as one Jev
        request, each turn carrying its `role`, `content` and
        `tools_called`; see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=[
                *self._required_test_case_params,
                MultiTurnParams.TOOLS_CALLED,
            ],
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

    def print_goals_and_steps_taken(self, goals_and_steps):
        final_goals_and_steps = ""
        for goal_step in goals_and_steps:
            final_goals_and_steps += f"{goal_step.user_goal} \n"
            final_goals_and_steps += (
                f"c{prettify_list(goal_step.steps_taken)} \n\n"
            )
        return final_goals_and_steps

    @property
    def __name__(self):
        return "Goal Accuracy"
