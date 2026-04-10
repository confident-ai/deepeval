from __future__ import annotations

from re import Pattern
from typing import List

from openai import AsyncOpenAI, OpenAI

from deepeval.metrics import BaseMetric
from deepeval.metrics.api import metric_data_manager
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.pollux.pollux_utils import (
    build_pollux_prompt,
    normalize_rubrics,
    parse_feedback,
    parse_score,
)
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.utils import get_or_create_event_loop


class PolluxJudgeMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        criteria_name: str,
        rubrics: dict[int | str, str],
        *,
        judge_model: str = "ai-forever/Pollux-4B-Judge",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "NONE",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        threshold: float = 0.5,
        normalize_score: bool = True,
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
        verbose_mode: bool = False,
        score_pattern: Pattern[str] | None = None,
        feedback_pattern: Pattern[str] | None = None,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        rubrics_text, rubric_keys = normalize_rubrics(rubrics)

        # Public names must match __init__ parameters so deepeval.metrics.utils.copy_metrics
        # can reconstruct this metric during evaluate() / async runs.
        self.criteria_name = criteria_name
        self.rubrics = rubrics
        self.judge_model = judge_model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.normalize_score = normalize_score
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.score_pattern = score_pattern
        self.feedback_pattern = feedback_pattern

        self._rubrics_text = rubrics_text
        self._rubric_keys = rubric_keys

        if strict_mode:
            self.threshold = (
                1.0 if normalize_score else float(max(self._rubric_keys))
            )
        else:
            self.threshold = threshold

        self.evaluation_model = judge_model
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    def _get_sync_client(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._sync_client

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key
            )
        return self._async_client

    def _normalize_pollux_score(self, raw_score: float) -> float:
        if not self.normalize_score:
            return raw_score

        min_key = float(self._rubric_keys[0])
        max_key = float(self._rubric_keys[-1])
        if max_key <= min_key:
            raise ValueError("rubrics must contain at least two score levels")

        normalized = (raw_score - min_key) / (max_key - min_key)
        return max(0.0, min(1.0, normalized))

    def _extract_test_case_fields(
        self, test_case: LLMTestCase
    ) -> tuple[str, str, str | None]:
        instruction = test_case.input
        answer = test_case.actual_output or ""
        reference_answer = test_case.expected_output
        return instruction, answer, reference_answer

    def _build_verbose_logs(
        self,
        prompt: str,
        raw: str,
        raw_score: float,
        feedback: str,
    ) -> None:
        if not self.verbose_mode:
            return

        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Prompt: {prompt[:200]}...",
                f"Raw response: {raw[:200]}...",
                f"Parsed score: {raw_score}",
                f"Feedback: {feedback[:200]}",
                f"Final score: {self.score}, Threshold: {self.threshold}",
            ],
        )

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
            None,
            test_case.multimodal,
        )

        self.evaluation_cost = None
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
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
                return self.score

            instruction, answer, reference_answer = (
                self._extract_test_case_fields(test_case)
            )
            prompt = build_pollux_prompt(
                instruction=instruction,
                answer=answer,
                criteria_name=self.criteria_name,
                rubrics=self._rubrics_text,
                reference_answer=reference_answer,
            )

            try:
                resp = self._get_sync_client().chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                raw = resp.choices[0].message.content or ""
                raw_score = parse_score(raw, pattern=self.score_pattern)
                feedback = parse_feedback(raw, pattern=self.feedback_pattern)
                if raw_score is None:
                    self.error = (
                        "Failed to parse score from judge response: "
                        f"{raw[:200]}"
                    )
                    raise ValueError(self.error)

                self.score = self._normalize_pollux_score(raw_score)
                self.reason = feedback if self.include_reason else None
                self.success = self.score >= self.threshold
                self._build_verbose_logs(prompt, raw, raw_score, feedback)
                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )
                return self.score
            except Exception as e:
                self.error = str(e)
                raise

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
            None,
            test_case.multimodal,
        )

        self.evaluation_cost = None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            instruction, answer, reference_answer = (
                self._extract_test_case_fields(test_case)
            )
            prompt = build_pollux_prompt(
                instruction=instruction,
                answer=answer,
                criteria_name=self.criteria_name,
                rubrics=self._rubrics_text,
                reference_answer=reference_answer,
            )

            try:
                resp = await self._get_async_client().chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                raw = resp.choices[0].message.content or ""
                raw_score = parse_score(raw, pattern=self.score_pattern)
                feedback = parse_feedback(raw, pattern=self.feedback_pattern)
                if raw_score is None:
                    self.error = (
                        "Failed to parse score from judge response: "
                        f"{raw[:200]}"
                    )
                    raise ValueError(self.error)

                self.score = self._normalize_pollux_score(raw_score)
                self.reason = feedback if self.include_reason else None
                self.success = self.score >= self.threshold
                self._build_verbose_logs(prompt, raw, raw_score, feedback)
                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )
                return self.score
            except Exception as e:
                self.error = str(e)
                raise

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
        return "Pollux Judge"
