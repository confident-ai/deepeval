from typing import Optional

from deepeval.metrics.voice.audio_level.volume_variance.utils import (
    VOLUME_MIN_WINDOWS,
    volume_drift_db,
)
from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class VolumeVarianceMetric(BaseVoiceMetric):
    # 10 dB is roughly half (or twice) as loud.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_drift_db: int = 10,
        flaky: bool = False,
    ):
        if max_drift_db <= 0:
            raise ValueError("max_drift_db must be greater than 0.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.max_drift_db = max_drift_db

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        measurements, skipped = agent_reply_measurements(test_case)
        eligible_turns = []
        defects = []
        for turn in measurements:
            drift = volume_drift_db(test_case.turns[turn].audio)
            if drift is None:
                skipped.append(
                    {
                        "turn": turn,
                        "error": f"Reply has under {VOLUME_MIN_WINDOWS} "
                        "seconds of speech to follow.",
                    }
                )
                continue
            eligible_turns.append(turn)
            if abs(drift) > self.max_drift_db:
                defects.append(
                    {
                        "type": "volume_drift",
                        "turn": turn,
                        "drift_db": round(drift, 1),
                    }
                )
        return defect_share_result(
            eligible_turns,
            defects,
            skipped,
            f"a volume drift over {self.max_drift_db} dB",
            no_eligible_reason="No assistant reply was long enough to "
            "follow its volume.",
        )

    @property
    def __name__(self):
        return "Volume Variance"
