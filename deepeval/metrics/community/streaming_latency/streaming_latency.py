from datetime import datetime, timezone
from typing import Dict, List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


def _parse_time(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        try:
            iso = text
            if iso.endswith("Z"):
                iso = iso[:-1] + "+00:00"
            moment = datetime.fromisoformat(iso)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return moment.timestamp()
        except ValueError:
            return None
    return None


class StreamingLatencyMetric(BaseMetric):
    """How fast does the response start, and how fast does it stream?

    Reads token timestamps from the trace's LLM span
    (``token_intervals``) — users notice time to first token and
    inter-token gaps more than total latency. Each configured budget
    scores ``1.0`` inside the budget and degrades proportionally
    outside it (``budget / observed`` for times, ``observed / minimum``
    for throughput); the metric score is the worst sub-score. Fully
    **deterministic**: no LLM, no API key, zero token cost.

    Time to first token needs the span's ``start_time``. Real nested
    trace dicts strip span timings, so pass ``token_times`` (and
    ``start_time``) explicitly when they are unavailable — see below.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        max_ttft: Optional[float] = None,
        max_tbt_mean: Optional[float] = None,
        min_tokens_per_sec: Optional[float] = None,
        threshold: Optional[float] = 1.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if (
            max_ttft is None
            and max_tbt_mean is None
            and min_tokens_per_sec is None
        ):
            raise ValueError(
                "StreamingLatencyMetric requires at least one of "
                "`max_ttft`, `max_tbt_mean` or `min_tokens_per_sec`."
            )
        for name, value in (
            ("max_ttft", max_ttft),
            ("max_tbt_mean", max_tbt_mean),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"`{name}` must be positive, got {value}.")
        if min_tokens_per_sec is not None and min_tokens_per_sec <= 0:
            raise ValueError(
                "`min_tokens_per_sec` must be positive, got "
                f"{min_tokens_per_sec}."
            )
        self.max_ttft = max_ttft
        self.max_tbt_mean = max_tbt_mean
        self.min_tokens_per_sec = min_tokens_per_sec
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.requires_trace = True
        # Deterministic metric: no evaluation model is used.
        self.model = None
        self.using_native_model = False
        self.async_mode = False

    def measure(
        self,
        test_case: LLMTestCase,
        token_times: Optional[List[float]] = None,
        start_time: Optional[float] = None,
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
            times, observed_start = self._resolve_times(
                test_case, token_times, start_time
            )
            stats = self._stats(times, observed_start)
            subscores = self._subscores(stats)
            if not subscores:
                self.score = 0.0
                self.success = False
                self.reason = self._missing_reason(times)
                self.verbose_logs = ""
                return self.score
            score = min(subscores.values())
            self.score_breakdown = {
                "ttft_s": stats["ttft"] or 0.0,
                "tbt_mean_s": stats["tbt_mean"] or 0.0,
                "tokens_per_sec": stats["tokens_per_sec"] or 0.0,
                "n_tokens": float(len(times)),
            }
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(stats, subscores)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Token timestamps: {len(times)}",
                    f"TTFT: {stats['ttft']}",
                    f"Mean time between tokens: {stats['tbt_mean']}",
                    f"Tokens per second: {stats['tokens_per_sec']}",
                    f"Sub-scores: {subscores}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        token_times: Optional[List[float]] = None,
        start_time: Optional[float] = None,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # Deterministic metric — no async work to do; reuse the sync path.
        return self.measure(
            test_case,
            token_times=token_times,
            start_time=start_time,
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

    def _resolve_times(self, test_case, token_times, start_time):
        if token_times is not None:
            parsed = [
                parsed_time
                for parsed_time in (_parse_time(t) for t in token_times)
                if parsed_time is not None
            ]
            return sorted(parsed), _parse_time(start_time)
        if test_case._trace_dict is None:
            return [], None
        for span in self._extract_all_spans(test_case._trace_dict):
            if str(span.get("type")).lower() != "llm":
                continue
            intervals = span.get("token_intervals")
            if intervals is None:
                intervals = span.get("tokenTimes")
            if not intervals:
                continue
            parsed = [
                parsed_time
                for parsed_time in (_parse_time(t) for t in intervals)
                if parsed_time is not None
            ]
            if parsed:
                observed = _parse_time(span.get("start_time"))
                if observed is None:
                    observed = _parse_time(span.get("startTime"))
                if start_time is not None:
                    observed = _parse_time(start_time)
                return sorted(parsed), observed
        return [], None

    @staticmethod
    def _stats(times: List[float], observed_start: Optional[float]):
        ordered = sorted(times)
        count = len(ordered)
        ttft = None
        if count and observed_start is not None:
            ttft = max(0.0, ordered[0] - observed_start)
        tbt_mean = None
        tokens_per_sec = None
        if count >= 2:
            gaps = [
                later - earlier for earlier, later in zip(ordered, ordered[1:])
            ]
            span = ordered[-1] - ordered[0]
            tbt_mean = sum(gaps) / len(gaps) if gaps else None
            if span > 0:
                tokens_per_sec = (count - 1) / span
        return {
            "ttft": ttft,
            "tbt_mean": tbt_mean,
            "tokens_per_sec": tokens_per_sec,
        }

    def _subscores(self, stats) -> Dict[str, float]:
        subscores: Dict[str, float] = {}
        if self.max_ttft is not None:
            if stats["ttft"] is None:
                subscores["ttft"] = 0.0
            elif stats["ttft"] <= self.max_ttft:
                subscores["ttft"] = 1.0
            else:
                subscores["ttft"] = max(0.0, self.max_ttft / stats["ttft"])
        if self.max_tbt_mean is not None and stats["tbt_mean"] is not None:
            if stats["tbt_mean"] <= self.max_tbt_mean:
                subscores["tbt"] = 1.0
            else:
                subscores["tbt"] = max(
                    0.0, self.max_tbt_mean / stats["tbt_mean"]
                )
        if (
            self.min_tokens_per_sec is not None
            and stats["tokens_per_sec"] is not None
        ):
            if stats["tokens_per_sec"] >= self.min_tokens_per_sec:
                subscores["throughput"] = 1.0
            else:
                subscores["throughput"] = max(
                    0.0, stats["tokens_per_sec"] / self.min_tokens_per_sec
                )
        return subscores

    def _missing_reason(self, times: List[float]) -> Optional[str]:
        if not self.include_reason:
            return None
        if not times:
            return (
                "No token timestamps found. This metric needs "
                "`token_intervals` on the trace's LLM span, or "
                "`token_times` passed to `measure`."
            )
        return (
            "Token timestamps exist but none of the configured "
            "budgets can be evaluated from them."
        )

    def _generate_reason(self, stats, subscores) -> Optional[str]:
        if not self.include_reason:
            return None
        parts = []
        if "ttft" in subscores:
            if stats["ttft"] is None:
                parts.append("time to first token is not measurable")
            else:
                parts.append(f"time to first token {stats['ttft']:.3f}s")
        if "tbt" in subscores and stats["tbt_mean"] is not None:
            parts.append(f"mean gap {stats['tbt_mean']:.3f}s")
        if "throughput" in subscores and stats["tokens_per_sec"] is not None:
            parts.append(f"{stats['tokens_per_sec']:.1f} tokens/s")
        detail = "; ".join(parts) if parts else "no measurable budgets"
        if min(subscores.values()) == 1.0:
            return f"Streaming latency is within budget ({detail})."
        return f"Streaming latency exceeds a budget ({detail})."

    @property
    def __name__(self):
        return "Streaming Latency"
