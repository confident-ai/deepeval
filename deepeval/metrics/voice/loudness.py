from typing import Optional

from deepeval.metrics.voice._defects import (
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class LoudnessMetric(BaseVoiceMetric):
    # Phone speech sits near -22 dBFS on this scale: quiet is 10 dB (half as
    # loud) below it, loud leaves under 6 dB of headroom before peaks clip.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        min_dbfs: int = -32,
        max_dbfs: int = -6,
        flaky: bool = False,
    ):
        if min_dbfs >= max_dbfs:
            raise ValueError("min_dbfs must be lower than max_dbfs.")
        if max_dbfs > 0:
            raise ValueError("max_dbfs must be 0 or lower.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.min_dbfs = min_dbfs
        self.max_dbfs = max_dbfs

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        measurements, skipped = agent_reply_measurements(test_case)
        defects = []
        for turn, reply in measurements.items():
            if reply.speech_dbfs < self.min_dbfs:
                event_type = "too_quiet"
            elif reply.speech_dbfs > self.max_dbfs:
                event_type = "too_loud"
            else:
                continue
            defects.append(
                {
                    "type": event_type,
                    "turn": turn,
                    "speech_dbfs": round(reply.speech_dbfs, 1),
                }
            )
        return defect_share_result(
            list(measurements),
            defects,
            skipped,
            f"speech outside {self.min_dbfs} to {self.max_dbfs} dBFS",
        )

    @property
    def __name__(self):
        return "Loudness"
