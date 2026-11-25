from typing import Optional

from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
    audible_replies,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class LoudnessMetric(BaseDefectMetric):
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

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        measurements, skipped = agent_reply_measurements(test_case)
        measurements = audible_replies(measurements, skipped)
        defects = []
        for turn, reply in measurements.items():
            if reply.speech_dbfs < self.min_dbfs:
                event_type = "too_quiet"
            elif reply.speech_dbfs > self.max_dbfs:
                event_type = "too_loud"
            else:
                continue
            defects.append(
                VoiceEvent(
                    type=event_type,
                    turn=turn,
                    speech_dbfs=round(reply.speech_dbfs, 1),
                    description=f"Reply was {event_type.replace('_', ' ')} "
                    f"({reply.speech_dbfs:.0f} dBFS)",
                )
            )
        return DefectShare(
            list(measurements),
            defects,
            skipped,
            f"speech outside {self.min_dbfs} to {self.max_dbfs} dBFS",
        )

    @property
    def __name__(self):
        return "Loudness"
