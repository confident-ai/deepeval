from typing import Optional

from deepeval.metrics.voice._analysis import (
    PITCH_MIN_FRAMES,
    PITCH_MIN_VOICED_SECONDS,
)
from deepeval.metrics.voice._defects import (
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class PitchVariationMetric(BaseVoiceMetric):
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

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
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
                {
                    "type": event_type,
                    "turn": turn,
                    "pitch_spread_semitones": round(
                        reply.pitch_spread_semitones, 1
                    ),
                }
            )
        return defect_share_result(
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
