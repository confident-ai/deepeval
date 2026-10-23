from typing import List, Optional, Tuple, Union

from deepeval.errors import MissingTestCaseParamsError
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from deepeval.metrics import BaseMetric
from deepeval.metrics.base_metric import Verdict
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    prepare_measure,
    construct_verbose_logs,
    initialize_model,
    initialize_system_one_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    SystemOneBinarySpec,
    SystemOneEvalSpec,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    system_one_probability,
    a_system_one_probability,
    format_decision_reason,
    verdict_from_probability,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.community.thought_action_consistency.template import (
    ThoughtActionConsistencyTemplate,
)
from deepeval.metrics.community.thought_action_consistency.schema import (
    ThoughtActionConsistencyVerdict,
    StepConsistencyVerdict,
)


class ThoughtActionConsistencyMetric(BaseMetric):
    """Per-step thought/action consistency for ReAct-style agent traces.

    For every tool call in ``tools_called`` that carries a ``reasoning``
    (the model's stated intent for that step), the judge checks whether the
    executed call — tool choice and arguments — actually carries out that
    intent. The score is the fraction of judged steps that are consistent.

    This is an *internal* consistency check and needs no reference:
    - ``ToolCorrectnessMetric`` compares calls against a golden
      ``expected_tools`` list (external expectation);
    - ``PlanAdherenceMetric`` compares execution against an expected plan
      (external expectation);
    - ``StepEfficiencyMetric`` returns one holistic judge score for the whole
      trace (aggregate, no per-step attribution).
    None of them flag the step-local failure this metric targets: the
    reasoning says one thing, the call does another.

    Tool calls without ``reasoning`` are skipped rather than judged, because
    routed/deterministic calls have no thought to compare against. A test case
    where *no* call carries reasoning raises ``MissingTestCaseParamsError``.

    ``score`` is ``consistent_steps / judged_steps``; with ``strict_mode`` the
    threshold becomes ``1.0`` and any shortfall scores ``0``.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.TOOLS_CALLED,
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

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case, multimodal=False)
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
                    return self._finalize_system_one_score()

                judged_steps = self._judged_steps(test_case)
                self.verdict = self._generate_verdict(judged_steps)
                self.score = self._calculate_score(judged_steps)
                self.reason = self._generate_reason()
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=self._verbose_steps(
                        judged_steps, len(test_case.tools_called or [])
                    ),
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case, multimodal=False)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self._finalize_system_one_score()

            judged_steps = self._judged_steps(test_case)
            self.verdict = await self._a_generate_verdict(judged_steps)
            self.score = self._calculate_score(judged_steps)
            self.reason = self._generate_reason()
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=self._verbose_steps(
                    judged_steps, len(test_case.tools_called or [])
                ),
            )
            return self.score

    ###############################################################
    # Pipeline
    ###############################################################

    def _judged_steps(
        self, test_case: LLMTestCase
    ) -> List[Tuple[int, ToolCall]]:
        """The steps that carry a reasoning, with their original trace index.

        Returns ``(original_index, call)`` pairs so verdicts and the reason
        map back to the exact positions in the user's ``tools_called`` list,
        even when calls without reasoning are skipped. Raises when no call
        carries a reasoning — there is nothing to compare against.
        """
        calls = test_case.tools_called or []
        judged = [
            (index, call)
            for index, call in enumerate(calls, start=1)
            if call.reasoning
        ]
        if not judged:
            raise MissingTestCaseParamsError(
                "No tool call in `tools_called` carries a `reasoning`. "
                "ThoughtActionConsistencyMetric compares each tool call's "
                "reasoning against its execution, so at least one call must "
                "have `ToolCall.reasoning` set. If your trace has no reasoning "
                "steps, use ToolCorrectnessMetric with `expected_tools` "
                "instead."
            )
        return judged

    def _build_prompt(self, judged_steps: List[Tuple[int, ToolCall]]) -> str:
        return ThoughtActionConsistencyTemplate.generate_verdict(
            steps=ThoughtActionConsistencyTemplate.format_steps(judged_steps),
        )

    def _generate_verdict(
        self, judged_steps: List[Tuple[int, ToolCall]]
    ) -> ThoughtActionConsistencyVerdict:
        p = system_one_probability(
            self, self._experimental_system_one_spec(judged_steps)
        )
        if p is not None:
            return self._system_one_verdict(p, judged_steps)
        prompt = self._build_prompt(judged_steps)
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ThoughtActionConsistencyVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: ThoughtActionConsistencyVerdict(**data),
        )

    async def _a_generate_verdict(
        self, judged_steps: List[Tuple[int, ToolCall]]
    ) -> ThoughtActionConsistencyVerdict:
        p = await a_system_one_probability(
            self, self._experimental_system_one_spec(judged_steps)
        )
        if p is not None:
            return self._system_one_verdict(p, judged_steps)
        prompt = self._build_prompt(judged_steps)
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ThoughtActionConsistencyVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: ThoughtActionConsistencyVerdict(**data),
        )

    def _experimental_system_one_spec(
        self, judged_steps: List[Tuple[int, ToolCall]]
    ) -> Optional[SystemOneBinarySpec]:
        return SystemOneBinarySpec(
            instructions=ThoughtActionConsistencyTemplate._experimental_system_one_verdict(),
            state={
                "steps": ThoughtActionConsistencyTemplate.format_steps(
                    judged_steps
                )
            },
        )

    def _system_one_verdict(
        self,
        p: float,
        judged_steps: List[Tuple[int, ToolCall]],
    ) -> ThoughtActionConsistencyVerdict:
        """System-one binary judgement: all steps consistent, or none.

        The binary system-one question asks whether every judged step is
        consistent. All steps share the same verdict, with a probability-based
        reason. The LLM schema path below still gives per-step attribution.
        """
        consistent = verdict_from_probability(p) == Verdict.YES
        return ThoughtActionConsistencyVerdict(
            verdicts=[
                StepConsistencyVerdict(
                    step_index=index,
                    verdict="consistent" if consistent else "inconsistent",
                    reasoning=format_decision_reason(
                        self, "P(all steps consistent)", p
                    ),
                )
                for index, _call in judged_steps
            ]
        )

    def _system_one_eval_spec(
        self, judged_steps: List[Tuple[int, ToolCall]]
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        the judged steps; see EXPERIMENTAL.md."""
        return SystemOneEvalSpec(
            evaluation_params=[SingleTurnParams.TOOLS_CALLED],
            questions=parse_questions(
                ThoughtActionConsistencyTemplate._experimental_system_one_questions()
            ),
            extra_state={
                "steps": ThoughtActionConsistencyTemplate.format_steps(
                    judged_steps
                )
            },
        )

    def _finalize_system_one_score(self) -> float:
        """Jev's score is a weighted mean of 0-1 question scores, which for
        this metric is already the fraction of judged steps that are
        consistent — no rounding is needed (unlike pass/fail metrics)."""
        raw = self.score
        self.score = raw
        if self.strict_mode and self.score < self.threshold:
            self.score = 0
        if self.reason is not None:
            self.reason += f"\nScore from Jev: {raw:.2f}"
        self.success = self.is_successful()
        return self.score

    ###############################################################
    # Score / reason
    ###############################################################

    def _calculate_score(
        self, judged_steps: List[Tuple[int, ToolCall]]
    ) -> float:
        verdict_by_index = {v.step_index: v for v in self.verdict.verdicts}
        consistent = 0
        for original_index, _call in judged_steps:
            verdict = verdict_by_index.get(original_index)
            if verdict is not None and verdict.verdict.strip().lower() == (
                "consistent"
            ):
                consistent += 1
        score = consistent / len(judged_steps)
        return 0 if self.strict_mode and score < self.threshold else score

    def _inconsistent_verdicts(self) -> List[StepConsistencyVerdict]:
        return [
            v
            for v in self.verdict.verdicts
            if v.verdict.strip().lower() == "inconsistent"
        ]

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None
        inconsistent = self._inconsistent_verdicts()
        if not self.verdict.verdicts:
            return (
                "The judge returned no verdicts, so no step was scored. "
                "This is an anomaly — please report it."
            )
        if not inconsistent:
            return "Every tool call executes the intent its reasoning declared."
        lines = [
            f"Steps inconsistent with their reasoning: {len(inconsistent)}."
        ]
        for verdict in inconsistent:
            detail = f" — {verdict.reasoning}" if verdict.reasoning else ""
            lines.append(f"Step {verdict.step_index}{detail}")
        return "\n".join(lines)

    def _verbose_steps(
        self,
        judged_steps: List[Tuple[int, ToolCall]],
        total_call_count: int,
    ) -> List[str]:
        verdict_lines = "\n".join(
            (
                f"Step {v.step_index}: {v.verdict} ({v.reasoning})"
                if v.reasoning
                else f"Step {v.step_index}: {v.verdict}"
            )
            for v in self.verdict.verdicts
        )
        return [
            f"Judged {len(judged_steps)} of {total_call_count} steps "
            f"with reasoning.",
            f"Verdicts:\n{verdict_lines}",
            f"Score: {self.score}\nReason: {self.reason}",
        ]

    @property
    def __name__(self):
        return "Thought Action Consistency"
