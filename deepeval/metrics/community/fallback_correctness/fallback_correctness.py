from typing import Dict, List, Optional, Union

from deepeval.metrics import BaseMetric
from deepeval.metrics.community.fallback_correctness.schema import (
    FallbackCorrectnessVerdict,
)
from deepeval.metrics.community.fallback_correctness.template import (
    FallbackCorrectnessTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    check_llm_test_case_params,
    construct_verbose_logs,
    generate_with_schema_and_extract,
    initialize_model,
    print_tools_called,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.utils import get_or_create_event_loop


class FallbackCorrectnessMetric(BaseMetric):
    """Evaluate whether an agent responds correctly when execution cannot finish.

    The ``context`` field describes the known failure or constraint, such as a
    tool timeout, an empty retrieval result, missing user information, or
    insufficient evidence. The metric judges whether ``actual_output``:

    1. acknowledges the relevant limitation,
    2. avoids unsupported claims or a false success claim, and
    3. takes an appropriate recovery action.

    Each satisfied criterion contributes one third of the score. The default
    threshold of ``1.0`` therefore requires all three criteria to pass.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.CONTEXT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 1.0,
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

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        self._validate_test_case(test_case)
        self._reset_evaluation_tracking()

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
                self.verdict = self._generate_verdict(test_case)
                self._set_result()

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        self._validate_test_case(test_case)
        self._reset_evaluation_tracking()

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.verdict = await self._a_generate_verdict(test_case)
            self._set_result()
            return self.score

    def _validate_test_case(self, test_case: LLMTestCase) -> None:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )

    def _reset_evaluation_tracking(self) -> None:
        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None

    def _set_result(self) -> None:
        self.score_breakdown = self._get_score_breakdown()
        score = sum(self.score_breakdown.values()) / len(self.score_breakdown)
        self.score = 0 if self.strict_mode and score < self.threshold else score
        self.reason = self._generate_reason()
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Verdict:\n{self.verdict.model_dump_json(indent=2)}",
                f"Score Breakdown:\n{self.score_breakdown}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    def _get_score_breakdown(self) -> Dict[str, float]:
        return {
            "acknowledges_limitation": float(
                self.verdict.acknowledges_limitation
            ),
            "avoids_unsupported_claims": float(
                self.verdict.avoids_unsupported_claims
            ),
            "recovery_action_appropriate": float(
                self.verdict.recovery_action_appropriate
            ),
        }

    def _build_prompt(self, test_case: LLMTestCase) -> str:
        fallback_context = "\n".join(
            f"- {context_item}" for context_item in test_case.context
        )
        tools_called = print_tools_called(test_case.tools_called)
        if tools_called == "":
            tools_called = "None provided."

        return FallbackCorrectnessTemplate.generate_verdict(
            input=test_case.input,
            fallback_context=fallback_context,
            tools_called=tools_called,
            actual_output=test_case.actual_output,
        )

    async def _a_generate_verdict(
        self, test_case: LLMTestCase
    ) -> FallbackCorrectnessVerdict:
        prompt = self._build_prompt(test_case)
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=FallbackCorrectnessVerdict,
            extract_schema=lambda schema: schema,
            extract_json=lambda data: FallbackCorrectnessVerdict(**data),
        )

    def _generate_verdict(
        self, test_case: LLMTestCase
    ) -> FallbackCorrectnessVerdict:
        prompt = self._build_prompt(test_case)
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=FallbackCorrectnessVerdict,
            extract_schema=lambda schema: schema,
            extract_json=lambda data: FallbackCorrectnessVerdict(**data),
        )

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None
        if self.verdict.reasoning:
            return self.verdict.reasoning

        failed_criteria = [
            name for name, score in self.score_breakdown.items() if score == 0
        ]
        if not failed_criteria:
            return "The response handles the fallback correctly."
        return "The response fails: " + ", ".join(failed_criteria) + "."

    @property
    def __name__(self):
        return "Fallback Correctness"
