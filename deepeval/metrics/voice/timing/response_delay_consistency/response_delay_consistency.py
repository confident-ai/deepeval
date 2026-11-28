from typing import Optional

from deepeval.metrics.voice.utils.call_activity import (
    EVENT_KEYS,
    reply_delays,
    wait_span,
)
from deepeval.metrics.voice.utils.defects import (
    MIN_COMPARED_REPLIES,
    DefectShare,
    shifts_from_typical,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import RANGE, VoiceEvent


class ResponseDelayConsistencyMetric(BaseDefectMetric):
    # A reply 2 s slower or faster than usual is as noticeable as a long
    # wait.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_shift_seconds: int = 2,
        use_speech_detection: bool = True,
        flaky: bool = False,
    ):
        if max_shift_seconds <= 0:
            raise ValueError("max_shift_seconds must be greater than 0.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.max_shift_seconds = max_shift_seconds
        self.use_speech_detection = use_speech_detection

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        delays, skipped, untimed_reason, source = reply_delays(
            test_case, self.use_speech_detection
        )
        event_key = EVENT_KEYS[source]
        defects = [
            VoiceEvent(
                type="delay_shift",
                **{event_key: key},
                **wait_span(
                    test_case,
                    event_key,
                    key,
                    delays[key],
                    self.use_speech_detection,
                ),
                response_delay_seconds=round(delays[key], 2),
                shift_seconds=round(shift, 2),
                mark=RANGE,
                description=f"Reply came {abs(shift):.1f}s "
                f"{'slower' if shift > 0 else 'faster'} than usual",
            )
            for key, shift in shifts_from_typical(delays).items()
            if abs(shift) > self.max_shift_seconds
        ]
        return DefectShare(
            list(delays),
            defects,
            skipped,
            f"a reply delay over {self.max_shift_seconds} seconds off the "
            "agent's typical delay",
            no_eligible_reason=untimed_reason
            or f"At least {MIN_COMPARED_REPLIES} timed assistant replies are "
            "needed to compare their delays.",
            min_eligible_turns=MIN_COMPARED_REPLIES,
            event_key=event_key,
            timing_source=source,
        )

    @property
    def __name__(self):
        return "Response Delay Consistency"
