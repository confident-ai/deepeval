from typing import Optional

from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.metrics.voice.utils.call_activity import (
    AgentReplies,
    agent_replies,
)
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.analysis import (
    PITCH_FRAME_MS,
    PITCH_JUMP_ABOVE_RANGE_SEMITONES,
    PITCH_RANGE_TOP_PERCENTILE,
    PITCH_STEP_RANGE_PERCENTILE,
    percentile,
    semitones,
)
from deepeval.metrics.voice.utils.events import MOMENT, VoiceEvent, placed

PITCH_FRAME_SECONDS = PITCH_FRAME_MS / 1000


class AbruptPitchJumpMetric(BaseDefectMetric):
    event_question = (
        "Does the speaker's voice suddenly jump to a different pitch, as if "
        "it glitched or cracked?"
    )
    event_speaker = "assistant"

    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None,
        eval_mode: Optional[VoiceEvalModeName] = None,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self._use_audio_decision_model(audio_decision_model, eval_mode)

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        replies = agent_replies(test_case)
        measurements, skipped = agent_reply_measurements(test_case)
        voiced = [
            pitch
            for reply in measurements.values()
            for pitch in reply.voiced_pitches_hz
        ]
        eligible_turns = []
        defects = []
        for turn, reply in measurements.items():
            if not reply.voiced_pitch_frames:
                skipped.append(
                    {
                        **replies.key(turn),
                        "error": "Reply has no voiced speech.",
                    }
                )
                continue
            eligible_turns.append(turn)
            if self.audio_decision_model is not None:
                # A stressed word steps up just like a glitch does, and only
                # a listener can tell them apart: with an audio decision
                # model, each step up into the top of the voice's range is a
                # moment for it to hear.
                defects.extend(
                    self._jump(replies, turn, at, "Pitch jumped abruptly")
                    for at, landing_hz in zip(
                        reply.pitch_step_times, reply.pitch_step_landings_hz
                    )
                    if landing_hz
                    >= percentile(voiced, PITCH_STEP_RANGE_PERCENTILE)
                )
                continue
            for at, landing_hz in zip(
                reply.pitch_jump_times, reply.pitch_jump_landings_hz
            ):
                above_range = semitones(
                    landing_hz, percentile(voiced, PITCH_RANGE_TOP_PERCENTILE)
                )
                if above_range < PITCH_JUMP_ABOVE_RANGE_SEMITONES:
                    continue
                defects.append(
                    self._jump(
                        replies,
                        turn,
                        at,
                        "Pitch jumped abruptly to "
                        f"{above_range:.0f} semitones above the voice's "
                        "usual range",
                        pitch_hz=round(landing_hz, 1),
                    )
                )
        return DefectShare(
            eligible_turns,
            defects,
            skipped,
            "an abrupt pitch jump",
            event_key=replies.event_key,
            timing_source=replies.timing_source,
            no_eligible_reason="No assistant reply had voiced speech to "
            "follow its pitch.",
        )

    @staticmethod
    def _jump(
        replies: AgentReplies,
        turn: float,
        at: float,
        description: str,
        **measured,
    ) -> VoiceEvent:
        # The step sits between two 40 ms frames; mark both.
        return VoiceEvent(
            type="pitch_jump",
            **replies.key(turn),
            mark=MOMENT,
            description=description,
            **measured,
            **placed(
                replies.audio[turn],
                max(0.0, at - PITCH_FRAME_SECONDS),
                2 * PITCH_FRAME_SECONDS,
            ),
        )

    @property
    def __name__(self):
        return "Abrupt Pitch Jump"
