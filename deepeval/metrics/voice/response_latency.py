from typing import Optional

from deepeval.metrics.voice._call_activity import EVENT_KEYS, reply_delays
from deepeval.metrics.voice._defects import defect_share_result
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class ResponseLatencyMetric(BaseVoiceMetric):
    # Voice agents need time to transcribe, think and speak; on human-labelled
    # phone calls, 4 s matched reviewers best (2 s flagged almost every call).
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_latency_seconds: int = 4,
        flaky: bool = False,
    ):
        if max_latency_seconds <= 0:
            raise ValueError("max_latency_seconds must be greater than 0.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.max_latency_seconds = max_latency_seconds

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        delays, skipped, untimed_reason, source = reply_delays(test_case)
        event_key = EVENT_KEYS[source]
        defects = [
            {
                "type": "slow_response",
                event_key: key,
                "response_delay_seconds": round(delay, 2),
            }
            for key, delay in delays.items()
            if delay > self.max_latency_seconds
        ]
        return defect_share_result(
            list(delays),
            defects,
            skipped,
            f"a wait over {self.max_latency_seconds} seconds before replying",
            no_eligible_reason=untimed_reason
            or "No assistant reply followed a timed caller turn.",
            event_key=event_key,
            timing_source=source,
        )

    @property
    def __name__(self):
        return "Response Latency"
