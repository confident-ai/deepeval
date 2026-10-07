import asyncio
from typing import Dict, List, Optional, Tuple, Union

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    initialize_model,
    check_llm_test_case_params,
    generate_rubric_score,
    a_generate_rubric_score,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.g_eval import schema as gschema
from deepeval.metrics.g_eval.utils import calculate_weighted_summed_score
from deepeval.config.settings import get_settings

from .utils import (
    JudgeEvalMessage,
    JudgeEvalVariable,
    render_messages,
    resolve_variables,
    validate_messages,
    validate_score_range,
    validate_variables,
)


class JudgeEval(BaseMetric):
    def __init__(
        self,
        name: str,
        messages: List[JudgeEvalMessage],
        variables: Optional[Dict[str, JudgeEvalVariable]] = None,
        score_range: Tuple[int, int] = (0, 10),
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: Optional[float] = 0.5,
        top_logprobs: int = 20,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        _include_judge_eval_suffix: bool = True,
    ):
        self.name = name
        self.messages = validate_messages(messages)
        self.variables = validate_variables(self.messages, variables)
        self.score_range = validate_score_range(score_range)
        self.score_range_span = self.score_range[1] - self.score_range[0]
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.threshold = 1 if strict_mode else threshold
        self.top_logprobs = top_logprobs
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self._include_judge_eval_suffix = _include_judge_eval_suffix

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _additional_context: Optional[str] = None,
    ) -> float:
        self._validate(test_case)

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                coro = self.a_measure(
                    test_case,
                    _show_indicator=False,
                    _in_component=_in_component,
                )
                settings = get_settings()
                loop.run_until_complete(
                    asyncio.wait_for(
                        coro,
                        timeout=(
                            None
                            if settings.DEEPEVAL_DISABLE_TIMEOUTS
                            else settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS
                        ),
                    )
                )
            else:
                prompt = self._results_prompt(test_case)
                judge_score, reason = generate_rubric_score(
                    metric=self,
                    prompt=prompt,
                    schema_cls=gschema.ReasonScore,
                    strict_mode=self.strict_mode,
                    top_logprobs=self.top_logprobs,
                    weighted_score_fn=calculate_weighted_summed_score,
                )
                self._set_result(prompt, judge_score, reason)

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _additional_context: Optional[str] = None,
    ) -> float:
        self._validate(test_case)

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            prompt = self._results_prompt(test_case)
            judge_score, reason = await a_generate_rubric_score(
                metric=self,
                prompt=prompt,
                schema_cls=gschema.ReasonScore,
                strict_mode=self.strict_mode,
                top_logprobs=self.top_logprobs,
                weighted_score_fn=calculate_weighted_summed_score,
            )
            self._set_result(prompt, judge_score, reason)
            return self.score

    def _validate(self, test_case: LLMTestCase) -> None:
        check_llm_test_case_params(
            test_case, [], None, None, self, self.model, test_case.multimodal
        )

    def _results_prompt(self, test_case: LLMTestCase) -> str:
        values = resolve_variables(test_case, self.variables)
        return self._get_prompt(
            "generate_evaluation_results",
            prompt=render_messages(self.messages, values),
            score_range=self.score_range,
            strict_mode=self.strict_mode,
        )

    def _set_result(
        self, prompt: str, judge_score: Union[int, float], reason: str
    ) -> None:
        normalized = (
            float(judge_score) - self.score_range[0]
        ) / self.score_range_span
        self.score = (
            (1 if normalized >= 1 else 0)
            if self.strict_mode
            else min(max(normalized, 0.0), 1.0)
        )
        self.success = self.is_successful()
        self.reason = reason
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Prompt:\n{prompt}",
                f"Score Range: {self.score_range[0]} to {self.score_range[1]}",
                f"Score: {self.score}",
                f"Reason: {self.reason}",
            ],
        )

    @property
    def __name__(self):
        if self._include_judge_eval_suffix:
            return f"{self.name} [JudgeEval]"
        return self.name
