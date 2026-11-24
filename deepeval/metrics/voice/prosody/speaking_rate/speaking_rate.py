from typing import Optional

from deepeval.metrics.voice.prosody.speaking_rate.utils import (
    speech_span_rate_wpm,
)
from deepeval.metrics.voice.utils.analysis import SPEECH_SPAN_MIN_SECONDS
from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent

SPEED_WORDS = {"too_slow": "too slowly", "too_fast": "too fast"}


class SpeakingRateMetric(BaseDefectMetric):
    # Conversation runs about 120-190 words per minute; each limit leaves a
    # 10 wpm margin.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        min_wpm: int = 110,
        max_wpm: int = 200,
        flaky: bool = False,
    ):
        if min_wpm < 0:
            raise ValueError("min_wpm must be 0 or more.")
        if min_wpm >= max_wpm:
            raise ValueError("min_wpm must be lower than max_wpm.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.min_wpm = min_wpm
        self.max_wpm = max_wpm

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        measurements, skipped = agent_reply_measurements(test_case)
        eligible_turns = []
        defects = []
        for turn, reply in measurements.items():
            rate = speech_span_rate_wpm(test_case.turns[turn].content, reply)
            if rate is None:
                skipped.append(
                    {
                        "turn": turn,
                        "error": "Reply has no text or under "
                        f"{SPEECH_SPAN_MIN_SECONDS:g} seconds of speech.",
                    }
                )
                continue
            eligible_turns.append(turn)
            if rate < self.min_wpm:
                event_type = "too_slow"
            elif rate > self.max_wpm:
                event_type = "too_fast"
            else:
                continue
            defects.append(
                VoiceEvent(
                    type=event_type,
                    turn=turn,
                    speech_rate_wpm=round(rate),
                    description=f"Agent spoke {SPEED_WORDS[event_type]} "
                    f"({rate:.0f} wpm)",
                )
            )
        return DefectShare(
            eligible_turns,
            defects,
            skipped,
            f"a speaking rate outside {self.min_wpm} to {self.max_wpm} "
            "words per minute",
            no_eligible_reason="No assistant reply had both text and enough "
            "speech to measure its speaking rate.",
        )

    @property
    def __name__(self):
        return "Speaking Rate"
