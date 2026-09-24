from typing import Dict, Optional, List, Union, Type

from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
    serialize_to_json,
)
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    initialize_system_one_model,
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
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.step_efficiency.schema import Task
from deepeval.metrics.plan_adherence.schema import (
    AgentPlan,
    PlanAdherenceScore,
)
from deepeval.templates import make_template_class


PlanAdherenceTemplate = make_template_class("PlanAdherenceMetric")

PLAN_ADHERENCE_LEVELS = [
    "No adherence",
    "Weak adherence",
    "Partial adherence",
    "Strong adherence",
    "Perfect adherence",
]


class PlanAdherenceMetric(BaseMetric):

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
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
        evaluation_template: Type[
            PlanAdherenceTemplate
        ] = PlanAdherenceTemplate,
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
        self.requires_trace = True
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ):
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

                task = self._extract_task_from_trace(test_case)
                agent_plan = self._extract_plan_from_trace(test_case)
                if len(agent_plan.plan) == 0:
                    self.score = 1
                    self.reason = "There were no plans to evaluate within the trace of your agent's execution. Please check if the agent's planning or reasoning or thinking is stored in any one of the trace attributes."
                else:
                    plan_adherence_score = self._get_plan_adherence_score(
                        task, agent_plan.plan, test_case
                    )
                    self.score = (
                        0
                        if self.strict_mode
                        and plan_adherence_score.score < self.threshold
                        else plan_adherence_score.score
                    )
                    self.reason = plan_adherence_score.reason
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Task: {task} \n",
                        f"Agent Plan: \n{prettify_list(agent_plan.plan)} \n",
                        f"Final Score: {self.score} \n",
                        f"Final Reason: {self.reason} \n",
                    ],
                )

                return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ):
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

            task = await self._a_extract_task_from_trace(test_case)
            agent_plan = await self._a_extract_plan_from_trace(test_case)
            if len(agent_plan.plan) == 0:
                self.score = 1
                self.reason = "There were no plans to evaluate within the trace of your agent's execution. Please check if the agent's planning or reasoning or thinking is stored in any one of the trace attributes."
            else:
                plan_adherence_score = await self._a_get_plan_adherence_score(
                    task, agent_plan.plan, test_case
                )
                self.score = (
                    0
                    if self.strict_mode
                    and plan_adherence_score.score < self.threshold
                    else plan_adherence_score.score
                )
                self.reason = plan_adherence_score.reason
            self.success = self.is_successful()

            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Task: {task} \n",
                    f"Agent Plan: \n{prettify_list(agent_plan.plan)} \n",
                    f"Final Score: {self.score} \n",
                    f"Final Reason: {self.reason} \n",
                ],
            )

            return self.score

    def _get_plan_adherence_score(self, task, plan, test_case):
        value = system_one_score(
            self, self._system_one_score_spec(task, plan, test_case)
        )
        if value is not None:
            return PlanAdherenceScore(
                score=value,
                reason=format_decision_reason(self, "plan adherence", value),
            )
        execution_trace_json = serialize_to_json(
            test_case._trace_dict, indent=2
        )
        prompt = self._get_prompt(
            "evaluate_adherence",
            template_class="PlanAdherenceMetric",
            user_task=task,
            agent_plan="\n".join(plan),
            execution_trace_json=execution_trace_json,
            multimodal=test_case.multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=PlanAdherenceScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: PlanAdherenceScore(**data),
        )

    async def _a_get_plan_adherence_score(self, task, plan, test_case):
        value = await a_system_one_score(
            self, self._system_one_score_spec(task, plan, test_case)
        )
        if value is not None:
            return PlanAdherenceScore(
                score=value,
                reason=format_decision_reason(self, "plan adherence", value),
            )
        execution_trace_json = serialize_to_json(
            test_case._trace_dict, indent=2
        )
        prompt = self._get_prompt(
            "evaluate_adherence",
            template_class="PlanAdherenceMetric",
            user_task=task,
            agent_plan="\n".join(plan),
            execution_trace_json=execution_trace_json,
            multimodal=test_case.multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=PlanAdherenceScore,
            extract_schema=lambda s: s,
            extract_json=lambda data: PlanAdherenceScore(**data),
        )

    def _extract_plan_from_trace(self, test_case: LLMTestCase) -> AgentPlan:
        trace_json_str = serialize_to_json(test_case._trace_dict, indent=2)
        prompt = self._get_prompt(
            "extract_plan_from_trace",
            template_class="PlanAdherenceMetric",
            trace_json_str=trace_json_str,
            multimodal=test_case.multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=AgentPlan,
            extract_schema=lambda s: s,
            extract_json=lambda data: AgentPlan(**data),
        )

    async def _a_extract_plan_from_trace(
        self, test_case: LLMTestCase
    ) -> AgentPlan:
        trace_json_str = serialize_to_json(test_case._trace_dict, indent=2)
        prompt = self._get_prompt(
            "extract_plan_from_trace",
            template_class="PlanAdherenceMetric",
            trace_json_str=trace_json_str,
            multimodal=test_case.multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=AgentPlan,
            extract_schema=lambda s: s,
            extract_json=lambda data: AgentPlan(**data),
        )

    def _extract_task_from_trace(self, test_case: LLMTestCase) -> str:
        trace_json = serialize_to_json(test_case._trace_dict, indent=2)
        prompt = self._get_prompt(
            "extract_task_from_trace",
            template_class="StepEfficiencyMetric",
            trace_json=trace_json,
            multimodal=test_case.multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Task,
            extract_schema=lambda s: s.task,
            extract_json=lambda data: data["task"],
        )

    async def _a_extract_task_from_trace(self, test_case: LLMTestCase) -> str:
        trace_json = serialize_to_json(test_case._trace_dict, indent=2)
        prompt = self._get_prompt(
            "extract_task_from_trace",
            template_class="StepEfficiencyMetric",
            trace_json=trace_json,
            multimodal=test_case.multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Task,
            extract_schema=lambda s: s.task,
            extract_json=lambda data: data["task"],
        )

    def _system_one_score_spec(
        self, task, plan, test_case: LLMTestCase
    ) -> Optional[SystemOneScoreSpec]:
        if test_case.multimodal or not isinstance(test_case._trace_dict, Dict):
            return None
        return SystemOneScoreSpec(
            instructions=self._get_prompt("_experimental_system_one_score"),
            levels=PLAN_ADHERENCE_LEVELS,
            state={
                "task": task,
                "plan": plan,
                "trace": compact_trace(test_case._trace_dict),
            },
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        the trace (or `input`, `actual_output` and `tools_called` when there
        is none); every question is not applicable when the agent states no
        plan, which scores 1 like the LLM chain; see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        has_trace = isinstance(test_case._trace_dict, Dict)
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
                )
            ),
            extra_state={
                "trace": (
                    compact_trace(test_case._trace_dict) if has_trace else None
                ),
            },
        )

    @property
    def __name__(self):
        return "Plan Adherence"
