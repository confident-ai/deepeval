from typing import Optional

from deepeval.metrics.voice.utils.analysis import NOISE_MIN_SECONDS
from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
    audible_replies,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class BackgroundNoiseMetric(BaseDefectMetric):
    event_question = "Is there distracting background noise behind the speaker?"
    event_speaker = "assistant"

    # 5 dB above the +15 dB speech-to-noise minimum for intelligible speech.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        min_snr_db: int = 20,
        audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None,
        eval_mode: Optional[VoiceEvalModeName] = None,
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
        self._use_audio_decision_model(audio_decision_model, eval_mode)
        self.min_snr_db = min_snr_db

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        measurements, skipped = agent_reply_measurements(test_case)
        measurements = audible_replies(measurements, skipped)
        eligible_turns = []
        defects = []
        for turn, reply in measurements.items():
            if reply.duration < NOISE_MIN_SECONDS:
                skipped.append(
                    {
                        "turn": turn,
                        "error": f"Reply is under {NOISE_MIN_SECONDS:g} "
                        "seconds, too short to measure its noise.",
                    }
                )
                continue
            eligible_turns.append(turn)
            if reply.estimated_snr_db < self.min_snr_db:
                defects.append(
                    VoiceEvent(
                        type="background_noise",
                        turn=turn,
                        estimated_snr_db=round(reply.estimated_snr_db, 1),
                        description="Background noise was high (SNR "
                        f"{reply.estimated_snr_db:.0f} dB)",
                    )
                )
        return DefectShare(
            eligible_turns,
            defects,
            skipped,
            f"background noise under {self.min_snr_db} dB signal-to-noise",
            no_eligible_reason="No assistant reply was long enough to "
            "measure its background noise.",
        )

    @property
    def __name__(self):
        return "Background Noise"
