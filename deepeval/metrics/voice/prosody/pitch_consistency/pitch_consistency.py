from typing import Optional

from deepeval.metrics.voice.utils.analysis import (
    PITCH_BASE_MIN_FRAMES,
    PITCH_BASE_MIN_VOICED_SECONDS,
    semitones,
)
from deepeval.metrics.voice.utils.defects import (
    MIN_COMPARED_REPLIES,
    agent_reply_measurements,
    DefectShare,
    shifts_from_typical,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class PitchConsistencyMetric(BaseDefectMetric):
    # Each reply is judged on its base pitch: the low end the voice keeps
    # returning to. On real calls that stays within 3 semitones of the
    # agent's other replies in all but about 1 reply in 1,000.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_shift_semitones: int = 3,
        flaky: bool = False,
    ):
        if max_shift_semitones <= 0:
            raise ValueError("max_shift_semitones must be greater than 0.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.max_shift_semitones = max_shift_semitones

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        measurements, skipped = agent_reply_measurements(test_case)
        levels = {}
        check_levels = {}
        for turn, reply in measurements.items():
            if reply.voiced_pitch_frames < PITCH_BASE_MIN_FRAMES:
                skipped.append(
                    {
                        "turn": turn,
                        "error": "Reply has under "
                        f"{PITCH_BASE_MIN_VOICED_SECONDS:g}s of voiced speech to "
                        "follow its pitch.",
                    }
                )
                continue
            levels[turn] = semitones(reply.pitch_base_hz, 1.0)
            check_levels[turn] = semitones(reply.pitch_base_check_hz, 1.0)
        # A shift only counts as far as both readings of the low end agree:
        # a few creaky frames move one of them, a changed voice moves both.
        check_shifts = shifts_from_typical(check_levels)
        shifts = {
            turn: (
                min(shift, check_shifts[turn], key=abs)
                if shift * check_shifts[turn] > 0
                else 0.0
            )
            for turn, shift in shifts_from_typical(levels).items()
        }
        defects = [
            VoiceEvent(
                type="pitch_shift",
                turn=turn,
                pitch_base_hz=round(measurements[turn].pitch_base_hz, 1),
                shift_semitones=round(shift, 1),
                description=f"Voice sat {abs(shift):.0f} semitones "
                f"{'higher' if shift > 0 else 'lower'} than in the agent's "
                "other replies",
            )
            for turn, shift in shifts.items()
            if abs(shift) > self.max_shift_semitones
        ]
        return DefectShare(
            list(levels),
            defects,
            skipped,
            f"a pitch over {self.max_shift_semitones} semitones off the "
            "agent's typical pitch",
            no_eligible_reason=f"At least {MIN_COMPARED_REPLIES} assistant "
            "replies with voiced speech are needed to compare their pitch.",
            min_eligible_turns=MIN_COMPARED_REPLIES,
        )

    @property
    def __name__(self):
        return "Pitch Consistency"
