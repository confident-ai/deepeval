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
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


MONOTONE_MIN_SECONDS = 4.0
EVENT_QUESTIONS = {
    "monotone": "Does the speaker sound flat and monotone, like a robot reading?",
    "erratic": "Does the speaker's pitch swing up and down erratically, in a way that sounds unnatural?",
}


class PitchVariationMetric(BaseDefectMetric):
    event_question = EVENT_QUESTIONS["monotone"]
    event_speaker = "assistant"

    # Adults vary their pitch by about 2.2 semitones (pitch sigma); under 1
    # is flatter than normal speech. A flat reply only sounds monotone once
    # it runs a few seconds. Expressive voices range widely without sounding
    # wrong, so there is no upper limit unless one is set.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        min_spread_semitones: int = 1,
        max_spread_semitones: Optional[int] = None,
        audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None,
        eval_mode: Optional[VoiceEvalModeName] = None,
        flaky: bool = False,
    ):
        if min_spread_semitones < 0:
            raise ValueError("min_spread_semitones must be 0 or more.")
        if (
            max_spread_semitones is not None
            and min_spread_semitones >= max_spread_semitones
        ):
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
        self._use_audio_decision_model(audio_decision_model, eval_mode)
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
            if (
                reply.pitch_spread_semitones < self.min_spread_semitones
                and reply.duration >= MONOTONE_MIN_SECONDS
            ):
                event_type = "monotone"
            elif (
                self.max_spread_semitones is not None
                and reply.pitch_spread_semitones > self.max_spread_semitones
            ):
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
            (
                f"pitch varying under {self.min_spread_semitones} semitones"
                if self.max_spread_semitones is None
                else f"pitch varying outside {self.min_spread_semitones} to "
                f"{self.max_spread_semitones} semitones"
            ),
            no_eligible_reason="No assistant reply had enough voiced speech "
            "to follow its pitch.",
        )

    def _event_question(self, event: VoiceEvent) -> Optional[str]:
        return EVENT_QUESTIONS[event.type]

    @property
    def __name__(self):
        return "Pitch Variation"
