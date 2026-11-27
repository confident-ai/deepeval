import re
from typing import Optional

from deepeval.metrics.voice.utils.analysis import SPEECH_SPAN_MIN_SECONDS
from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.metrics.voice.utils.call_activity import agent_replies
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent

SPEED_WORDS = {"too_slow": "too slowly", "too_fast": "too fast"}
# Three letters or digits in a row, each said on its own ("B, Y, R"), a
# letter given by a word ("B as in Boy"), or a code ("SH5216742").
SPELLED_OUT = re.compile(
    r"(?:(?<!['’.])\b\w\b[,\s-]+){2}\b\w\b(?!['’])"
    r"|\b\w as in \w+"
    r"|\b(?=\w*\d)\w{6,}\b"
)
EVENT_QUESTIONS = {
    "too_slow": "Is the speaker talking unnaturally slowly?",
    "too_fast": "Is the speaker talking unnaturally fast?",
}
# With an audio decision model to confirm each one, replies this much closer
# to a normal pace are also worth its hearing. The syllable count alone can't
# tell them from normal speech.
SECOND_CHECK_MARGIN = 1.5


def spells_something_out(content: Optional[str]) -> bool:
    return bool(content and SPELLED_OUT.search(content))


class SpeakingRateMetric(BaseDefectMetric):
    event_question = "Is the speaker talking at an unnatural speed?"
    event_speaker = "assistant"

    # Agents on real calls speak 5.1 to 7.7 syllables a second (the middle
    # 90% of replies); the limits sit beyond the slowest and fastest 1 in
    # 1,000.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        min_syllables_per_second: float = 3.5,
        max_syllables_per_second: float = 10.0,
        audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None,
        eval_mode: Optional[VoiceEvalModeName] = None,
        flaky: bool = False,
    ):
        if min_syllables_per_second < 0:
            raise ValueError("min_syllables_per_second must be 0 or more.")
        if min_syllables_per_second >= max_syllables_per_second:
            raise ValueError(
                "min_syllables_per_second must be lower than "
                "max_syllables_per_second."
            )
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self._use_audio_decision_model(audio_decision_model, eval_mode)
        self.min_syllables_per_second = min_syllables_per_second
        self.max_syllables_per_second = max_syllables_per_second

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        replies = agent_replies(test_case)
        measurements, skipped = agent_reply_measurements(test_case)
        margin = (
            SECOND_CHECK_MARGIN if self.audio_decision_model is not None else 0
        )
        eligible_turns = []
        defects = []
        for turn, reply in measurements.items():
            rate = reply.syllables_per_second
            if rate is None:
                skipped.append(
                    {
                        **replies.key(turn),
                        "error": "Reply has under "
                        f"{SPEECH_SPAN_MIN_SECONDS:g} seconds of speech.",
                    }
                )
                continue
            # A name or an email read out letter by letter is slow on
            # purpose.
            if replies.by_turn and spells_something_out(
                test_case.turns[turn].content
            ):
                skipped.append(
                    {
                        **replies.key(turn),
                        "error": "Reply spells something out.",
                    }
                )
                continue
            eligible_turns.append(turn)
            if rate < self.min_syllables_per_second + margin:
                event_type = "too_slow"
            elif rate > self.max_syllables_per_second - margin:
                event_type = "too_fast"
            else:
                continue
            defects.append(
                VoiceEvent(
                    type=event_type,
                    **replies.key(turn),
                    syllables_per_second=round(rate, 1),
                    description=f"Agent spoke {SPEED_WORDS[event_type]} "
                    f"({rate:.1f} syllables a second)",
                )
            )
        return DefectShare(
            eligible_turns,
            defects,
            skipped,
            f"a speaking rate outside {self.min_syllables_per_second:g} to "
            f"{self.max_syllables_per_second:g} syllables a second",
            event_key=replies.event_key,
            timing_source=replies.timing_source,
            no_eligible_reason="No assistant reply had enough speech to "
            "measure its speaking rate.",
        )

    def _event_question(self, event: VoiceEvent) -> Optional[str]:
        return EVENT_QUESTIONS[event.type]

    @property
    def __name__(self):
        return "Speaking Rate"
