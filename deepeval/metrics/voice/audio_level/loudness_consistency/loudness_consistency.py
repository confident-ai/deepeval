from typing import Optional

from deepeval.metrics.voice.utils.defects import (
    MIN_COMPARED_REPLIES,
    agent_reply_measurements,
    DefectShare,
    shifts_from_typical,
    audible_replies,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class LoudnessConsistencyMetric(BaseDefectMetric):
    # A 3 dB change is just perceptible; 5 dB is clearly noticeable.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_shift_db: int = 5,
        flaky: bool = False,
    ):
        if max_shift_db <= 0:
            raise ValueError("max_shift_db must be greater than 0.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.max_shift_db = max_shift_db

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        measurements, skipped = agent_reply_measurements(test_case)
        measurements = audible_replies(measurements, skipped)
        levels = {
            turn: reply.speech_dbfs for turn, reply in measurements.items()
        }
        defects = [
            VoiceEvent(
                type="loudness_shift",
                turn=turn,
                speech_dbfs=round(levels[turn], 1),
                shift_db=round(shift, 1),
                description=f"Reply was {abs(shift):.0f} dB "
                f"{'louder' if shift > 0 else 'quieter'} than usual",
            )
            for turn, shift in shifts_from_typical(levels).items()
            if abs(shift) > self.max_shift_db
        ]
        return DefectShare(
            list(levels),
            defects,
            skipped,
            f"a loudness over {self.max_shift_db} dB off the agent's "
            "typical loudness",
            no_eligible_reason=f"At least {MIN_COMPARED_REPLIES} decodable "
            "assistant replies are needed to compare their loudness.",
            min_eligible_turns=MIN_COMPARED_REPLIES,
        )

    @property
    def __name__(self):
        return "Loudness Consistency"
