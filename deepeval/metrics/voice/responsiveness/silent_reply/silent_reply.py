import difflib
from typing import Optional, Union

from deepeval.metrics.voice.speech_quality.transcription_accuracy.utils import (
    words_for_error_rate,
)
from deepeval.metrics.voice.utils.analysis import decode_audio, get_numpy
from deepeval.metrics.voice.utils.call_activity import (
    call_activity,
    read_call_recording,
    sound_between,
)
from deepeval.metrics.voice.utils.defects import (
    NO_CALL_AUDIO_REASON,
    agent_reply_measurements,
    call_has_audio,
    DefectShare,
    is_inaudible,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.models.base_model import DeepEvalBaseSTT
from deepeval.models.speech_selection import initialize_stt_model
from deepeval.test_case import Audio, ConversationalTestCase
from deepeval.voice.connectors.audio_utils import pcm16_to_wav_bytes
from deepeval.metrics.voice.utils.events import VoiceEvent


# A reply stamped over the caller's own speech holds no agent audio because
# the agent was listening, not because its reply was lost.
CALLER_SPEAKING_MIN_SECONDS = 0.5
# A reply stamped shorter than this, or too short to say its own text even at
# twice a fast talker's pace, was stamped in the wrong place: its silence
# says nothing about the reply.
STAMPED_REPLY_MIN_SECONDS = 1.0
STAMPED_REPLY_MAX_WORDS_PER_SECOND = 6.0
# With a speech-to-text model, the call this far either side of a silent
# reply is transcribed; a reply whose words are heard there, in order, was
# spoken, just not where it was stamped.
REPLY_SEARCH_SECONDS = 15.0
REPLY_HEARD_MIN_WORD_SHARE = 0.6


class SilentReplyMetric(BaseDefectMetric):
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

    def _spoken_nearby(
        self, test_case: ConversationalTestCase, turn: int
    ) -> bool:
        channels = read_call_recording(test_case.audio, test_case.agent_channel)
        audio = test_case.turns[turn].audio
        words = words_for_error_rate(test_case.turns[turn].content)
        if (
            self.stt_model is None
            or channels is None
            or audio.start_time is None
            or not words
        ):
            return False
        np = get_numpy()
        caller, agent, sample_rate = channels
        samples, turn_rate = decode_audio(audio)
        end = audio.start_time + len(samples) / turn_rate
        first = int(
            max(0.0, audio.start_time - REPLY_SEARCH_SECONDS) * sample_rate
        )
        last = min(
            len(caller),
            len(agent),
            int((end + REPLY_SEARCH_SECONDS) * sample_rate),
        )
        both = (
            caller[first:last].astype(np.int32)
            + agent[first:last].astype(np.int32)
        ) // 2
        heard, cost = self.stt_model.transcribe(
            Audio.from_bytes(
                pcm16_to_wav_bytes(
                    both.astype(np.int16).tobytes(), sample_rate
                ),
                "audio/wav",
                encoding="wav",
            )
        )
        self.evaluation_cost = (self.evaluation_cost or 0.0) + (cost or 0.0)
        # In order, so common words like "you" and "the" heard around the
        # reply don't pass for the reply itself.
        found = sum(
            block.size
            for block in difflib.SequenceMatcher(
                None, words, words_for_error_rate(heard), autojunk=False
            ).get_matching_blocks()
        )
        return found / len(words) >= REPLY_HEARD_MIN_WORD_SHARE

    def _stamp_is_off(
        self, test_case: ConversationalTestCase, turn: int
    ) -> bool:
        """The silent audio is not where the reply was: it is too short for
        the reply's text, or the caller was talking through it."""
        audio = test_case.turns[turn].audio
        samples, sample_rate = decode_audio(audio)
        seconds = len(samples) / sample_rate
        words = len(test_case.turns[turn].content.split())
        if words and (
            seconds < STAMPED_REPLY_MIN_SECONDS
            or words / seconds > STAMPED_REPLY_MAX_WORDS_PER_SECOND
        ):
            return True
        if audio.start_time is None:
            return False
        caller = call_activity(test_case, self.use_speech_detection).caller
        return (
            sound_between(
                caller,
                audio.start_time,
                audio.start_time + len(samples) / sample_rate,
            )
            >= CALLER_SPEAKING_MIN_SECONDS
        )

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        defect_label = "a reply the caller could not hear"
        if not call_has_audio(test_case):
            return DefectShare(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason=NO_CALL_AUDIO_REASON,
            )
        measurements, skipped = agent_reply_measurements(test_case)
        undecodable = {entry["turn"] for entry in skipped}
        eligible_turns = []
        defects = []
        for index, turn in enumerate(test_case.turns):
            if (
                turn.role != "assistant"
                or index in undecodable
                # A turn with no words had nothing to say, heard or not.
                or not turn.content.strip()
            ):
                continue
            if (
                turn.audio is not None
                and is_inaudible(measurements[index])
                and self._stamp_is_off(test_case, index)
            ):
                continue
            eligible_turns.append(index)
            if turn.audio is None:
                defects.append(
                    VoiceEvent(
                        type="no_audio",
                        turn=index,
                        description="Reply had no audio",
                    )
                )
            elif is_inaudible(measurements[index]) and not self._spoken_nearby(
                test_case, index
            ):
                defects.append(
                    VoiceEvent(
                        type="no_speech",
                        turn=index,
                        description="Reply had no audible speech",
                    )
                )
        return DefectShare(
            eligible_turns,
            defects,
            skipped,
            defect_label,
            no_eligible_reason="No assistant turn had text.",
        )

    @property
    def __name__(self):
        return "Silent Reply"
