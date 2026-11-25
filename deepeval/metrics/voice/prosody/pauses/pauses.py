from typing import Optional

from deepeval.metrics.voice.utils.analysis import SPEECH_SPAN_MIN_SECONDS
from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent, placed


class PausesMetric(BaseDefectMetric):
    # Long pauses in spontaneous speech cluster near 1.5 s; over 50% means
    # more silence than speech.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_pause_ms: int = 2000,
        max_pause_percent: int = 50,
        flaky: bool = False,
    ):
        if max_pause_ms <= 0:
            raise ValueError("max_pause_ms must be greater than 0.")
        if not 0 < max_pause_percent <= 100:
            raise ValueError("max_pause_percent must be between 1 and 100.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.max_pause_ms = max_pause_ms
        self.max_pause_percent = max_pause_percent

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        measurements, skipped = agent_reply_measurements(test_case)
        eligible_turns = []
        defects = []
        for turn, reply in measurements.items():
            if reply.speech_span_seconds < SPEECH_SPAN_MIN_SECONDS:
                skipped.append(
                    {
                        "turn": turn,
                        "error": "Reply has under "
                        f"{SPEECH_SPAN_MIN_SECONDS:g} seconds of speech.",
                    }
                )
                continue
            eligible_turns.append(turn)
            if reply.longest_pause_ms > self.max_pause_ms:
                defects.append(
                    VoiceEvent(
                        type="long_pause",
                        turn=turn,
                        longest_pause_ms=round(reply.longest_pause_ms),
                        description="Agent paused for "
                        f"{reply.longest_pause_ms / 1000:.1f}s mid-reply",
                        **placed(
                            test_case.turns[turn].audio,
                            *reply.longest_pause_span,
                        ),
                    )
                )
            elif reply.pause_fraction * 100 > self.max_pause_percent:
                defects.append(
                    VoiceEvent(
                        type="too_much_silence",
                        turn=turn,
                        pause_fraction=round(reply.pause_fraction, 2),
                        description="Reply was "
                        f"{reply.pause_fraction * 100:.0f}% silence",
                    )
                )
        return DefectShare(
            eligible_turns,
            defects,
            skipped,
            f"a pause over {self.max_pause_ms} ms or over "
            f"{self.max_pause_percent}% silence",
            no_eligible_reason="No assistant reply had enough speech to "
            "measure its pauses.",
        )

    @property
    def __name__(self):
        return "Pauses"
