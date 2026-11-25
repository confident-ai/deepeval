from typing import Optional

from deepeval.metrics.voice.utils.analysis import (
    PITCH_MIN_FRAMES,
    PITCH_MIN_VOICED_SECONDS,
)
from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class PitchVariationMetric(BaseDefectMetric):
    # Adults vary their pitch by about 2.2 semitones (pitch sigma): 1 is
    # flatter than normal speech, 6 nearly three times the norm.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        min_spread_semitones: int = 1,
        max_spread_semitones: int = 6,
        flaky: bool = False,
    ):
        if min_spread_semitones < 0:
            raise ValueError("min_spread_semitones must be 0 or more.")
        if min_spread_semitones >= max_spread_semitones:
            raise ValueError(
                "min_spread_semitones must be lower than max_spread_semitones."
            )
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.min_spread_semitones = min_spread_semitones
        self.max_spread_semitones = max_spread_semitones

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        measurements, skipped = agent_reply_measurements(test_case)
        eligible_turns = []
        defects = []
        for turn, reply in measurements.items():
            if reply.voiced_pitch_frames < PITCH_MIN_FRAMES:
                skipped.append(
                    {
                        "turn": turn,
                        "error": "Reply has under "
                        f"{PITCH_MIN_VOICED_SECONDS:g}s of voiced speech to "
                        "follow its pitch.",
                    }
                )
                continue
            eligible_turns.append(turn)
            if reply.pitch_spread_semitones < self.min_spread_semitones:
                event_type = "monotone"
            elif reply.pitch_spread_semitones > self.max_spread_semitones:
                event_type = "erratic"
            else:
                continue
            defects.append(
                VoiceEvent(
                    type=event_type,
                    turn=turn,
                    pitch_spread_semitones=round(
                        reply.pitch_spread_semitones, 1
                    ),
                    description=(
                        "Reply sounded monotone"
                        if event_type == "monotone"
                        else "Pitch varied erratically"
                    ),
                )
            )
        return DefectShare(
            eligible_turns,
            defects,
            skipped,
            f"pitch varying outside {self.min_spread_semitones} to "
            f"{self.max_spread_semitones} semitones",
            no_eligible_reason="No assistant reply had enough voiced speech "
            "to follow its pitch.",
        )

    @property
    def __name__(self):
        return "Pitch Variation"
