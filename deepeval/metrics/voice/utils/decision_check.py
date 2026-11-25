"""A second check on the events a voice metric flagged: an audio decision
model hears a short clip around each event, and events it judges not real
are dropped before scoring. A clip that can't be cut, or a check that fails,
keeps the event."""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

from deepeval.metrics.voice.utils.analysis import decode_audio
from deepeval.metrics.voice.utils.call_activity import read_call_recording
from deepeval.metrics.voice.utils.defects import DefectShare
from deepeval.metrics.voice.utils.events import VoiceEvent
from deepeval.models.system_one.schema import NoulQuestion
from deepeval.test_case import Audio, ConversationalTestCase
from deepeval.voice.connectors.audio_utils import pcm16_to_wav_bytes

if TYPE_CHECKING:
    from deepeval.metrics.voice.base_metric import BaseDefectMetric

# A yes share under this means most listeners would not call the event real.
EVENT_DISMISS_BELOW = 0.5
# Seconds of call heard around an event; the API takes 1 s to 5 min.
EVENT_CLIP_SECONDS = 5.0
MIN_CLIP_SECONDS = 1.0
MAX_CLIP_SECONDS = 300.0
MAX_PARALLEL_CHECKS = 4
PCM16_SAMPLE_WIDTH = 2
EVENT_CLIP_CONTEXT = (
    "A short clip from a phone call between an AI voice agent and a caller."
)
EVENT_QUESTION_KEY = "event"

EventSpeaker = Literal["user", "assistant"]

logger = logging.getLogger(__name__)


def confirm_defects(
    metric: "BaseDefectMetric",
    test_case: ConversationalTestCase,
    share: DefectShare,
) -> DefectShare:
    model = metric.audio_decision_model
    question = NoulQuestion(instructions=metric.event_question)
    recording = read_call_recording(test_case.audio, test_case.agent_channel)

    def check(defect: VoiceEvent) -> Tuple[bool, float]:
        if model is None:
            return True, 0.0
        try:
            clip = metric._event_clip(test_case, defect, recording)
            if clip is None:
                return True, 0.0
            answers, cost = model.decide(
                clip, {EVENT_QUESTION_KEY: question}, EVENT_CLIP_CONTEXT
            )
        except Exception as error:
            logger.warning(
                "%s kept a flagged event its audio decision check failed on: %s",
                metric.__name__,
                error,
            )
            return True, 0.0
        answer = answers.nouls.get(EVENT_QUESTION_KEY)
        real = answer is None or answer.probability >= EVENT_DISMISS_BELOW
        return real, cost or 0.0

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHECKS) as pool:
        verdicts = list(pool.map(check, share.defects))
    metric.evaluation_cost = (metric.evaluation_cost or 0.0) + sum(
        cost for _, cost in verdicts
    )
    kept = [
        defect for defect, (real, _) in zip(share.defects, verdicts) if real
    ]
    return replace(
        share, defects=kept, dismissed=len(share.defects) - len(kept)
    )


def event_clip(
    test_case: ConversationalTestCase,
    event: VoiceEvent,
    speaker: EventSpeaker,
    recording: Optional[Tuple[Any, Any, int]],
) -> Optional[bytes]:
    """The speaker's audio around the event: from the stereo call recording
    when there is one, else from the speaker's own turn."""
    if event.start_time is not None:
        start = event.start_time
        end = start + (event.duration_seconds or 0.0)
        if recording is not None:
            caller, agent, sample_rate = recording
            channel = caller if speaker == "user" else agent
            return _clip(channel.tobytes(), sample_rate, start, end)
        for turn in test_case.turns:
            audio = turn.audio
            if (
                turn.role != speaker
                or audio is None
                or audio.start_time is None
            ):
                continue
            decoded = _decoded(audio)
            if decoded is None:
                continue
            pcm, sample_rate = decoded
            turn_end = audio.start_time + _seconds(pcm, sample_rate)
            if audio.start_time <= end and turn_end >= start:
                return _clip(
                    pcm,
                    sample_rate,
                    start - audio.start_time,
                    end - audio.start_time,
                )
        return None
    if event.turn is None or not 0 <= event.turn < len(test_case.turns):
        return None
    if test_case.turns[event.turn].role != speaker:
        return None
    if event.offset_seconds is None:
        return turn_clip(test_case, event.turn)
    audio = test_case.turns[event.turn].audio
    decoded = _decoded(audio) if audio is not None else None
    if decoded is None:
        return None
    pcm, sample_rate = decoded
    return _clip(
        pcm,
        sample_rate,
        event.offset_seconds,
        event.offset_seconds + (event.duration_seconds or 0.0),
    )


def turn_clip(
    test_case: ConversationalTestCase, turn: Optional[int]
) -> Optional[bytes]:
    if turn is None or not 0 <= turn < len(test_case.turns):
        return None
    audio = test_case.turns[turn].audio
    decoded = _decoded(audio) if audio is not None else None
    if decoded is None:
        return None
    pcm, sample_rate = decoded
    return _clip(pcm, sample_rate, 0.0, _seconds(pcm, sample_rate))


def _decoded(audio: Audio) -> Optional[Tuple[bytes, int]]:
    try:
        samples, sample_rate = decode_audio(audio)
    except (TypeError, ValueError):
        return None
    return samples.tobytes(), sample_rate


def _seconds(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / PCM16_SAMPLE_WIDTH / sample_rate


def _clip(
    pcm: bytes, sample_rate: int, start: float, end: float
) -> Optional[bytes]:
    length = _seconds(pcm, sample_rate)
    start, end = max(0.0, start), min(length, max(end, start))
    if start >= length:
        return None
    grow = max(0.0, EVENT_CLIP_SECONDS - (end - start)) / 2
    start, end = max(0.0, start - grow), min(length, end + grow)
    end = min(end, start + MAX_CLIP_SECONDS)
    first = int(start * sample_rate) * PCM16_SAMPLE_WIDTH
    window = pcm[first : int(end * sample_rate) * PCM16_SAMPLE_WIDTH]
    floor = int(MIN_CLIP_SECONDS * sample_rate) * PCM16_SAMPLE_WIDTH
    if len(window) < floor:
        window += b"\x00" * (floor - len(window))
    return pcm16_to_wav_bytes(window, sample_rate)
