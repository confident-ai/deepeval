from typing import Optional

from deepeval.metrics.voice.audio_level.volume_variance.utils import (
    VOLUME_MIN_WINDOWS,
    volume_drift_db,
)
from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    audible_replies,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.metrics.voice.utils.call_activity import agent_replies
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class VolumeVarianceMetric(BaseDefectMetric):
    # 10 dB is roughly half (or twice) as loud. Speech moves that much on
    # its own from one second to the next, so only a volume that keeps
    # fading or swelling through the reply counts, and listeners only
    # minded fades past 15 dB.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_drift_db: int = 15,
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

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        replies = agent_replies(test_case, self.use_speech_detection)
        measurements, skipped = agent_reply_measurements(
            test_case, self.use_speech_detection
        )
        measurements = audible_replies(measurements, skipped, replies.event_key)
        eligible_turns = []
        defects = []
        for turn in measurements:
            drift = volume_drift_db(replies.audio[turn])
            if drift is None:
                skipped.append(
                    {
                        **replies.key(turn),
                        "error": f"Reply has under {VOLUME_MIN_WINDOWS} "
                        "seconds of speech to follow.",
                    }
                )
                continue
            eligible_turns.append(turn)
            if abs(drift) > self.max_drift_db:
                defects.append(
                    VoiceEvent(
                        type="volume_drift",
                        **replies.key(turn),
                        drift_db=round(drift, 1),
                        description="Volume "
                        f"{'faded' if drift < 0 else 'swelled'} "
                        f"{abs(drift):.0f} dB during the reply",
                    )
                )
        return replies.share(
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
