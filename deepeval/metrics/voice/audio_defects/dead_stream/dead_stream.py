import difflib
from typing import List, Optional, Union

from deepeval.metrics.voice.speech_quality.transcription_accuracy.utils import (
    words_for_error_rate,
)
from deepeval.metrics.voice.utils.analysis import get_numpy
from deepeval.metrics.voice.utils.call_activity import (
    agent_replies,
    call_activity,
    read_call_recording,
    sound_between,
)
from deepeval.metrics.voice.utils.defects import (
    agent_glitch_events,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.models.base_model import DeepEvalBaseSTT
from deepeval.models.speech_selection import initialize_stt_model
from deepeval.test_case import Audio, ConversationalTestCase
from deepeval.voice.connectors.audio_utils import pcm16_to_wav_bytes
from deepeval.metrics.voice.utils.events import VoiceEvent

# The agent going quiet while the caller talks is the agent listening, even
# when all the caller says is "yes".
CALLER_SPEAKING_MIN_SECONDS = 0.2
# With a speech-to-text model, the agent's audio from this long before a hole
# to this long after it is transcribed. A hole after a finished sentence is
# the agent pausing to look something up or waiting for an answer, not its
# audio dying; the speech after the hole lets punctuation tell the two apart.
WORDS_BEFORE_HOLE_SECONDS = 4.0
WORDS_AFTER_HOLE_SECONDS = 3.0
SENTENCE_ENDS = (".", "?", "!")
# Speech-to-text marks speech that trails off with an ellipsis.
TRAILING_OFF = ("...", "…")
CLOSING_MARKS = "\"'”’)]"


def dead_streams(
    test_case: ConversationalTestCase, use_speech_detection: bool
) -> List[VoiceEvent]:
    """The holes in the agent's replies: its audio goes dead and then carries
    on. Dead air to the end of a reply is left to Abrupt Cut-off."""
    _, events, _ = agent_glitch_events(test_case, use_speech_detection)
    holes = [
        event
        for event in events
        if event.type == "dead_stream" and not event.runs_to_end
    ]
    if not any(event.start_time is not None for event in holes):
        return holes
    caller = call_activity(test_case, use_speech_detection).caller
    return [
        event
        for event in holes
        if event.start_time is None
        or sound_between(
            caller,
            event.start_time,
            event.start_time + event.duration_seconds,
        )
        < CALLER_SPEAKING_MIN_SECONDS
    ]


def _ends_a_sentence(word: str) -> bool:
    word = word.rstrip(CLOSING_MARKS)
    return word.endswith(SENTENCE_ENDS) and not word.endswith(TRAILING_OFF)


class DeadStreamMetric(BaseDefectMetric):
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        use_speech_detection: bool = True,
        stt_model: Optional[Union[str, DeepEvalBaseSTT]] = None,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.use_speech_detection = use_speech_detection
        self.stt_model = (
            stt_model
            if stt_model is None or isinstance(stt_model, DeepEvalBaseSTT)
            else initialize_stt_model(stt_model)
        )

    def _transcribe(self, samples, sample_rate: int) -> str:
        np = get_numpy()
        heard, cost = self.stt_model.transcribe(
            Audio.from_bytes(
                pcm16_to_wav_bytes(
                    samples.astype(np.int16).tobytes(), sample_rate
                ),
                "audio/wav",
                encoding="wav",
            )
        )
        self.evaluation_cost = (self.evaluation_cost or 0.0) + (cost or 0.0)
        return heard

    def _after_a_finished_sentence(
        self, test_case: ConversationalTestCase, hole: VoiceEvent
    ) -> bool:
        channels = read_call_recording(test_case.audio, test_case.agent_channel)
        if (
            self.stt_model is None
            or channels is None
            or hole.start_time is None
        ):
            return False
        _, agent, sample_rate = channels
        first = int(
            max(0.0, hole.start_time - WORDS_BEFORE_HOLE_SECONDS) * sample_rate
        )
        stop = int(hole.start_time * sample_rate)
        last = int(
            (hole.start_time + hole.duration_seconds + WORDS_AFTER_HOLE_SECONDS)
            * sample_rate
        )
        before = words_for_error_rate(
            self._transcribe(agent[first:stop], sample_rate)
        )
        if not before:
            return False
        across = self._transcribe(agent[first:last], sample_rate).split()
        normalized = [
            next(iter(words_for_error_rate(word)), "") for word in across
        ]
        blocks = [
            block
            for block in difflib.SequenceMatcher(
                None, normalized, before, autojunk=False
            ).get_matching_blocks()
            if block.size
        ]
        if not blocks:
            return False
        # The last word heard before the hole, as written in the transcript
        # that runs on past it.
        last_block = max(blocks, key=lambda block: block.b + block.size)
        return _ends_a_sentence(across[last_block.a + last_block.size - 1])

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        replies = agent_replies(test_case, self.use_speech_detection)
        eligible_turns, _, skipped = agent_glitch_events(
            test_case, self.use_speech_detection
        )
        defects = [
            event.model_copy(
                update={
                    "description": "Agent's audio went dead for "
                    f"{event.duration_seconds:.1f}s"
                }
            )
            for event in dead_streams(test_case, self.use_speech_detection)
            if not self._after_a_finished_sentence(test_case, event)
        ]
        return replies.share(
            eligible_turns,
            defects,
            skipped,
            "a dead stream",
        )

    @property
    def __name__(self):
        return "Dead Stream"
