import json
from typing import Dict, List, Optional, Tuple

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


class ToolRetryThrashMetric(BaseMetric):
    """Is the agent repeating calls or flip-flopping between tools?

    Scores two deterministic sub-signals from the trace's tool spans
    (``@observe``) and averages them:

    1. **Tool Retry** — repeats of the same tool with identical
       arguments. Degrades to ``0.5`` at ``repetition_threshold``
       repeats and ``0.0`` at twice that.
    2. **Tool Thrash** — flip-flops (``A, B, A``) between tools.
       Scores ``0.5`` within ``alternation_threshold`` flip-flops
       and ``0.0`` beyond it.

    ``1.0`` means no repetition and no flip-flopping. Fully
    **deterministic**: no LLM, no API key, zero token cost.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        repetition_threshold: int = 3,
        alternation_threshold: int = 2,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if repetition_threshold < 2:
            raise ValueError(
                "`repetition_threshold` must be at least 2, got "
                f"{repetition_threshold}."
            )
        if alternation_threshold < 1:
            raise ValueError(
                "`alternation_threshold` must be at least 1, got "
                f"{alternation_threshold}."
            )
        self.repetition_threshold = repetition_threshold
        self.alternation_threshold = alternation_threshold
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        # Async only reuses the sync path; kept for house consistency.
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.requires_trace = True
        # Deterministic metric: no evaluation model is used.
        self.model = None
        self.using_native_model = False

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
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
        self.test_case = test_case
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if test_case._trace_dict is None:
                self.score = 0.0
                self.success = False
                self.reason = (
                    "No trace data found. This metric requires trace "
                    "data from @observe."
                )
                self.verbose_logs = ""
                return self.score

            tool_spans = [
                span
                for span in self._extract_all_spans(test_case._trace_dict)
                if str(span.get("type")).lower() == "tool"
            ]
            retry_score, retry_reason = self._score_retry(tool_spans)
            thrash_score, thrash_reason = self._score_thrash(tool_spans)
            score = (retry_score + thrash_score) / 2
            self.score_breakdown = {
                "retry_score": retry_score,
                "thrash_score": thrash_score,
            }
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            if score == 1.0:
                self.reason = (
                    "No tool repetition or flip-flopping detected."
                    if self.include_reason
                    else None
                )
            else:
                self.reason = (
                    f"{retry_reason} {thrash_reason}".strip()
                    if self.include_reason
                    else None
                )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Tools: {[s.get('name') for s in tool_spans]}",
                    f"Retry: {retry_score} ({retry_reason})",
                    f"Thrash: {thrash_score} ({thrash_reason})",
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

    @staticmethod
    def _call_key(span: Dict):
        name = span.get("name", "")
        input_val = span.get("input", {})
        if isinstance(input_val, str):
            try:
                input_val = json.loads(input_val)
            except Exception:
                pass
        if isinstance(input_val, dict):
            args = tuple(sorted((str(k), str(v)) for k, v in input_val.items()))
        else:
            args = (str(input_val),)
        return (name, args)

    def _score_retry(self, tool_spans: List[Dict]) -> Tuple[float, str]:
        if not tool_spans:
            return 1.0, "No tool spans found."
        counts: Dict[tuple, int] = {}
        for span in tool_spans:
            key = self._call_key(span)
            counts[key] = counts.get(key, 0) + 1
        top_key, top_count = max(counts.items(), key=lambda item: item[1])
        if top_count >= self.repetition_threshold * 2:
            return 0.0, (
                f"Tool '{top_key[0]}' repeated {top_count} times "
                "with identical arguments."
            )
        if top_count >= self.repetition_threshold:
            return 0.5, (
                f"Tool '{top_key[0]}' repeated {top_count} times "
                "with identical arguments."
            )
        return 1.0, "No excessive tool repetition."

    def _score_thrash(self, tool_spans: List[Dict]) -> Tuple[float, str]:
        names = [str(span.get("name", "")) for span in tool_spans]
        flips = sum(
            1
            for first, second, third in zip(names, names[1:], names[2:])
            if first != second and first == third
        )
        if flips == 0:
            return 1.0, "No flip-flopping between tools."
        if flips <= self.alternation_threshold:
            return 0.5, f"Flipped between tools {flips} time(s)."
        return 0.0, f"Flipped between tools {flips} time(s)."

    @property
    def __name__(self):
        return "Tool Retry Thrash"
