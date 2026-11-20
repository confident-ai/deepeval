from typing import Dict, List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


class ContextWindowBudgetMetric(BaseMetric):
    """Is the prompt dangerously close to the model's context limit?

    Divides prompt tokens by the model's context window
    (``max_window_tokens``). Prompt tokens come from
    ``LLMTestCase.input_token_count`` when set, otherwise from the
    summed ``input_token_count`` of the trace's LLM spans, otherwise
    from a ``chars / 4`` estimate (flagged in the reason). Scores
    ``1.0`` while usage stays under ``warn_threshold`` of the window
    (default 90%); above that the score decays proportionally, like
    ``LatencyBudgetMetric`` over its budget. Fully **deterministic**:
    no LLM, no API key, zero token cost.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
    ]

    def __init__(
        self,
        max_window_tokens: int,
        warn_threshold: float = 0.9,
        threshold: Optional[float] = 1.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if max_window_tokens <= 0:
            raise ValueError(
                "`max_window_tokens` must be positive, got "
                f"{max_window_tokens}."
            )
        if not 0 < warn_threshold <= 1:
            raise ValueError(
                "`warn_threshold` must be in (0, 1], got " f"{warn_threshold}."
            )
        self.max_window_tokens = max_window_tokens
        self.warn_threshold = warn_threshold
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        # Deterministic metric: no evaluation model is used.
        self.model = None
        self.using_native_model = False
        self.async_mode = False

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(
            test_case, self._required_params, None, None, self
        )
        self.test_case = test_case
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            prompt_tokens, estimated = self._prompt_tokens(test_case)
            budget = self.max_window_tokens * self.warn_threshold
            score = (
                1.0
                if prompt_tokens <= budget
                else max(0.0, budget / prompt_tokens)
            )
            utilization = prompt_tokens / self.max_window_tokens
            self.score_breakdown = {
                "utilization": utilization,
                "prompt_tokens": float(prompt_tokens),
                "window_tokens": float(self.max_window_tokens),
                "budget_tokens": float(budget),
            }
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(
                prompt_tokens, utilization, estimated
            )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Prompt tokens: {prompt_tokens}"
                    + (" (estimated)" if estimated else ""),
                    f"Window: {self.max_window_tokens}",
                    f"Utilization: {utilization:.2%}",
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
        # Deterministic metric — no async work to do; reuse the sync path.
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    @staticmethod
    def _extract_all_spans(trace_dict: Dict) -> List[Dict]:
        spans: List[Dict] = []

        def traverse(span: Dict):
            if span:
                spans.append(span)
                for child in span.get("children", []):
                    traverse(child)

        traverse(trace_dict)
        return spans

    def _prompt_tokens(self, test_case: LLMTestCase):
        if test_case.input_token_count is not None:
            return int(test_case.input_token_count), False
        if test_case._trace_dict is not None:
            total = sum(
                span.get("input_token_count", 0) or 0
                for span in self._extract_all_spans(test_case._trace_dict)
                if str(span.get("type")).lower() == "llm"
                and span.get("input_token_count") is not None
            )
            if total:
                return int(total), False
        estimate = max(1, len(test_case.input or "") // 4)
        return estimate, True

    def _generate_reason(
        self, prompt_tokens: int, utilization: float, estimated: bool
    ) -> Optional[str]:
        if not self.include_reason:
            return None
        source = "estimated prompt tokens" if estimated else "prompt tokens"
        if utilization <= self.warn_threshold:
            return (
                f"{prompt_tokens} {source} use {utilization:.1%} of the "
                f"{self.max_window_tokens}-token window, within the "
                f"{self.warn_threshold:.0%} budget."
            )
        return (
            f"{prompt_tokens} {source} use {utilization:.1%} of the "
            f"{self.max_window_tokens}-token window, above the "
            f"{self.warn_threshold:.0%} budget."
        )

    @property
    def __name__(self):
        return "Context Window Budget"
