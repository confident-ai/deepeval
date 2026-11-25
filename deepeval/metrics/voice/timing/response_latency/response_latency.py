from typing import Optional

from deepeval.metrics.voice.utils.call_activity import (
    EVENT_KEYS,
    reply_delays,
    wait_span,
)
from deepeval.metrics.voice.utils.defects import DefectShare
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import RANGE, VoiceEvent


class ResponseLatencyMetric(BaseDefectMetric):
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
        use_speech_detection: bool = True,
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
        self.use_speech_detection = use_speech_detection

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        delays, skipped, untimed_reason, source = reply_delays(
            test_case, self.use_speech_detection
        )
        event_key = EVENT_KEYS[source]
        defects = [
            VoiceEvent(
                type="slow_response",
                **{event_key: key},
                **wait_span(
                    test_case, event_key, key, delay, self.use_speech_detection
                ),
                response_delay_seconds=round(delay, 2),
                mark=RANGE,
                description=f"Agent took {delay:.1f}s to reply",
            )
            for key, delay in delays.items()
            if delay > self.max_latency_seconds
        ]
        return DefectShare(
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
