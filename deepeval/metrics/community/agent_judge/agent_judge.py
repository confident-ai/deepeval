from typing import Any, List, Optional, Union
import json

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    prepare_measure,
    construct_verbose_logs,
    initialize_model,
    generate_with_schema_and_extract,
    a_generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.community.agent_judge.schema import (
    AgentJudgment,
    AgentMilestones,
)


class AgentJudgeMetric(BaseMetric):
    """Does the agent's work satisfy the task's milestones?

    Implements the Agent-as-a-Judge framework (Zhuge et al.,
    "Agent-as-a-Judge: Evaluate Agents with Agents", arXiv:2410.10934,
    ICML 2025) for a single test case: the task ``input`` is decomposed
    into ordered milestones (or supplied up front), and an LLM judge
    issues a fresh, independent verdict per milestone against the
    agent's trajectory — final output plus the tool calls that produced
    it. The score is the mean milestone verdict, so partial progress
    yields a reward signal rather than a binary outcome.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        requirements: Optional[List[str]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        if requirements is not None:
            self.requirements = AgentJudgeMetric._normalize_requirements(
                requirements
            )
            if not self.requirements:
                raise ValueError(
                    "AgentJudgeMetric requires at least one non-empty "
                    "requirement when 'requirements' is provided."
                )
        else:
            self.requirements = None
        self.milestones: Optional[List[str]] = None
        self.milestone_scores: Optional[List[float]] = None

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case)
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
                milestones = self._resolve_milestones(test_case)
                judgment = self._judge(test_case, milestones)
                self._fill(test_case, milestones, judgment)
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            milestones = await self._a_resolve_milestones(test_case)
            judgment = await a_generate_with_schema_and_extract(
                metric=self,
                prompt=self._build_judgment_prompt(test_case, milestones),
                schema_cls=AgentJudgment,
                extract_schema=lambda result: result,
                extract_json=lambda data: AgentJudgment(**data),
            )
            self._fill(test_case, milestones, judgment)
            return self.score

    def _resolve_milestones(self, test_case: LLMTestCase) -> List[str]:
        if self.requirements:
            # Supplied requirements stand in for the benchmark's manual
            # annotations, so decomposition is skipped entirely.
            return list(self.requirements)
        milestones = generate_with_schema_and_extract(
            metric=self,
            prompt=self._build_milestones_prompt(test_case),
            schema_cls=AgentMilestones,
            extract_schema=lambda result: result,
            extract_json=lambda data: AgentMilestones(**data),
        ).milestones
        return self._normalize_milestones(milestones)

    async def _a_resolve_milestones(self, test_case: LLMTestCase) -> List[str]:
        if self.requirements:
            return list(self.requirements)
        milestones = await a_generate_with_schema_and_extract(
            metric=self,
            prompt=self._build_milestones_prompt(test_case),
            schema_cls=AgentMilestones,
            extract_schema=lambda result: result,
            extract_json=lambda data: AgentMilestones(**data),
        )
        return self._normalize_milestones(milestones.milestones)

    def _judge(
        self, test_case: LLMTestCase, milestones: List[str]
    ) -> AgentJudgment:
        return generate_with_schema_and_extract(
            metric=self,
            prompt=self._build_judgment_prompt(test_case, milestones),
            schema_cls=AgentJudgment,
            extract_schema=lambda result: result,
            extract_json=lambda data: AgentJudgment(**data),
        )

    def _fill(
        self,
        test_case: LLMTestCase,
        milestones: List[str],
        judgment: AgentJudgment,
    ):
        self.milestones = list(milestones)
        self.milestone_scores = AgentJudgeMetric._align_scores(
            judgment.scores, len(milestones)
        )
        score = (
            sum(self.milestone_scores) / len(self.milestone_scores)
            if self.milestone_scores
            else 0.0
        )
        score = min(1.0, max(0.0, score))
        self.score = 0 if self.strict_mode and score < self.threshold else score
        self.success = self.is_successful()
        self.reason = judgment.reason if self.include_reason else None
        verdict_lines = [
            f"{milestone}: {verdict:.1f}"
            for milestone, verdict in zip(milestones, self.milestone_scores)
        ]
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Input: {test_case.input}",
                f"Trajectory:\n{self._trajectory(test_case)}",
                "Milestones:\n" + "\n".join(verdict_lines),
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    @staticmethod
    def _normalize_requirements(requirements: List[str]) -> List[str]:
        return [
            requirement.strip()
            for requirement in (requirements or [])
            if requirement and requirement.strip()
        ]

    @staticmethod
    def _normalize_milestones(milestones: Optional[List[str]]) -> List[str]:
        normalized = AgentJudgeMetric._normalize_requirements(milestones)
        if not normalized:
            raise ValueError(
                "The evaluation LLM returned no usable milestones. "
                "Please use a better evaluation model, or pass explicit "
                "'requirements' to AgentJudgeMetric."
            )
        return normalized

    @staticmethod
    def _align_scores(scores: Optional[List[float]], count: int) -> List[float]:
        # Judges occasionally miscount milestones. Pad missing verdicts
        # with 0.0 and drop extras so the mean stays per-milestone, and
        # clamp every verdict into [0, 1].
        aligned = [
            min(1.0, max(0.0, float(score))) for score in (scores or [])[:count]
        ]
        aligned.extend([0.0] * (count - len(aligned)))
        return aligned

    @staticmethod
    def _trajectory(test_case: LLMTestCase) -> str:
        # Tool outputs can be dumps of entire files; cap each one so a
        # single call cannot crowd the rest of the evidence out of the
        # judge's prompt.
        max_output_chars = 400
        lines = [f"Final output: {test_case.actual_output}"]
        calls = test_case.tools_called or []
        if not calls:
            lines.append("Tool calls: none")
            return "\n".join(lines)
        lines.append("Tool calls:")
        for index, call in enumerate(calls, 1):
            args = (
                call.input_parameters
                if call.input_parameters is not None
                else {}
            )
            rendered_args = AgentJudgeMetric._dumps(args)
            output = call.output if call.output is not None else "no output"
            rendered_output = str(output)
            if len(rendered_output) > max_output_chars:
                rendered_output = rendered_output[:max_output_chars] + "..."
            lines.append(
                f"{index}. {call.name} args={rendered_args} "
                f"output={rendered_output}"
            )
        return "\n".join(lines)

    @staticmethod
    def _dumps(value: Any) -> str:
        try:
            return json.dumps(
                value, sort_keys=True, ensure_ascii=False, default=str
            )
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _build_milestones_prompt(test_case: LLMTestCase) -> str:
        return (
            "You decompose an agent task into ordered milestones for "
            "evaluation. Each milestone must be independently checkable "
            "against the agent's work. Reply with JSON only: "
            '{"milestones": ["<milestone 1>", "<milestone 2>", ...]}. '
            "Return between 2 and 6 milestones, ordered from most "
            "fundamental to most optional.\n"
            f"Task: {test_case.input}"
        )

    @staticmethod
    def _build_judgment_prompt(
        test_case: LLMTestCase, milestones: List[str]
    ) -> str:
        numbered = "\n".join(
            f"{index}. {milestone}"
            for index, milestone in enumerate(milestones, 1)
        )
        return (
            "You judge whether an agent's work satisfies each task "
            "milestone. Judge each milestone independently with fresh "
            "evidence from the trajectory — do not let one verdict "
            "influence another, and do not accept claims in the final "
            "output without support from the tool calls. Reply with "
            "JSON only: "
            '{"scores": [<0.0-1.0 per milestone in order>], '
            '"reason": "<one sentence>"}. Use 1.0 for fully met, 0.5 '
            "for partially met, 0.0 for not met.\n"
            f"Task: {test_case.input}\n"
            f"Milestones:\n{numbered}\n"
            f"Agent trajectory:\n{AgentJudgeMetric._trajectory(test_case)}"
        )

    @property
    def __name__(self):
        return "Agent Judge"
