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


class BackgroundNoiseMetric(BaseVoiceMetric):
    # Under about 20 dB signal-to-noise, noise behind speech is clearly audible.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        min_snr_db: int = 20,
        flaky: bool = False,
    ):
        if min_snr_db < 0:
            raise ValueError("min_snr_db must be 0 or more.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.min_snr_db = min_snr_db

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        measurements, skipped = agent_reply_measurements(test_case)
        defects = [
            {
                "type": "background_noise",
                "turn": turn,
                "estimated_snr_db": round(reply.estimated_snr_db, 1),
            }
            for turn, reply in measurements.items()
            if reply.estimated_snr_db < self.min_snr_db
        ]
        return defect_share_result(
            list(measurements),
            defects,
            skipped,
            f"background noise under {self.min_snr_db} dB signal-to-noise",
        )

    @property
    def __name__(self):
        return "Background Noise"
