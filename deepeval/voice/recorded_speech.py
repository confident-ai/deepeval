"""Play recorded caller lines, with voice events laid into them."""

from array import array
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from deepeval.models.base_model import DeepEvalBaseTTS
from deepeval.models.tts._frames import frame_pcm_stream
from deepeval.simulator.events import (
    BackgroundLine,
    DeadAir,
    Hold,
    MidSentencePause,
    NonSpeechSound,
    SideSpeech,
    SlowToAnswer,
    VoiceEvent,
)
from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.connectors import audio_utils

# Quiet after an aside or a "hang on" before the caller carries on.
CLIP_GAP_SECONDS = 0.4
# How far into the caller's line another speaker starts.
BACKGROUND_LINE_DELAY_SECONDS = 0.3

Synthesize = Callable[[str], Awaitable[Audio]]


def _pcm(audio: Audio, sample_rate: int) -> bytes:
    pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
    pcm = audio_utils.downmix_to_mono(pcm, channels)
    if rate != sample_rate:
        pcm = audio_utils.resample_pcm16(pcm, rate, sample_rate)
    return pcm


def _silence(seconds: float, sample_rate: int) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


def _scaled(pcm: bytes, gain: float) -> bytes:
    samples = array("h", pcm)
    return array(
        "h", (max(-32768, min(32767, int(sample * gain))) for sample in samples)
    ).tobytes()


def _mixed(pcm: bytes, over: bytes, start: int) -> bytes:
    samples = array("h", pcm)
    overlay = array("h", over)
    if start + len(overlay) > len(samples):
        samples.extend([0] * (start + len(overlay) - len(samples)))
    for index, sample in enumerate(overlay):
        mixed = samples[start + index] + sample
        samples[start + index] = max(-32768, min(32767, mixed))
    return samples.tobytes()


def _offset(event: VoiceEvent, text: str, line_seconds: float) -> float:
    """Seconds into the line an event falls."""
    if event.at_seconds is not None:
        return min(event.at_seconds, line_seconds)
    words = max(1, len(text.split()))
    after = event.after_words if event.after_words is not None else words // 2
    return line_seconds * min(after, words) / words


async def assemble_utterance(
    text: str,
    recording: Optional[Audio],
    events: List[VoiceEvent],
    sample_rate: int,
    synthesize: Synthesize,
) -> Tuple[Audio, List[Dict[str, Any]]]:
    """Build the audio for one caller turn and say where each event fell.

    Returns the audio and one record per event, with `offset_seconds` from
    the start of the audio and `seconds` for events that take time.
    """

    async def clip(event: VoiceEvent) -> Optional[bytes]:
        if event.audio is not None:
            return _pcm(event.audio, sample_rate)
        if getattr(event, "text", None):
            return _pcm(await synthesize(event.text), sample_rate)
        return None

    line = _pcm(recording or await synthesize(text), sample_rate)
    seconds_of = lambda pcm: len(pcm) / 2 / sample_rate
    line_seconds = seconds_of(line)

    # Sounds and pauses inside the line, placed latest first so earlier
    # offsets stay true while the line grows.
    inside: List[Tuple[float, VoiceEvent, bytes]] = []
    for event in events:
        if isinstance(event, MidSentencePause):
            inside.append(
                (
                    _offset(event, text, line_seconds),
                    event,
                    _silence(event.seconds, sample_rate),
                )
            )
        elif isinstance(event, NonSpeechSound) and event.audio is not None:
            inside.append(
                (
                    _offset(event, text, line_seconds),
                    event,
                    _pcm(event.audio, sample_rate),
                )
            )
    placed: Dict[int, Tuple[float, float]] = {}
    for at, event, piece in sorted(
        inside, key=lambda item: item[0], reverse=True
    ):
        cut = int(at * sample_rate) * 2
        line = line[:cut] + piece + line[cut:]
    shift = 0.0
    for at, event, piece in sorted(inside, key=lambda item: item[0]):
        placed[id(event)] = (at + shift, seconds_of(piece))
        shift += seconds_of(piece)

    for event in events:
        if isinstance(event, BackgroundLine):
            spoken = await clip(event)
            if spoken is not None:
                start = int(BACKGROUND_LINE_DELAY_SECONDS * sample_rate) * 2
                line = _mixed(line, _scaled(spoken, event.volume), start // 2)
                placed[id(event)] = (
                    BACKGROUND_LINE_DELAY_SECONDS,
                    seconds_of(spoken),
                )

    lead = b""
    for event in events:
        if isinstance(event, (Hold, DeadAir)):
            placed[id(event)] = (seconds_of(lead), event.seconds)
            lead += _silence(event.seconds, sample_rate)
        elif isinstance(event, (SlowToAnswer, SideSpeech)):
            spoken = await clip(event)
            if spoken is None:
                continue
            if isinstance(event, SideSpeech):
                spoken = _scaled(spoken, event.volume)
            placed[id(event)] = (seconds_of(lead), seconds_of(spoken))
            lead += spoken + _silence(CLIP_GAP_SECONDS, sample_rate)

    lead_seconds = seconds_of(lead)
    records = []
    for event in events:
        record = event.record()
        if id(event) in placed:
            at, seconds = placed[id(event)]
            in_line = not isinstance(
                event, (Hold, DeadAir, SlowToAnswer, SideSpeech)
            )
            record["offset_seconds"] = at + (lead_seconds if in_line else 0.0)
            record["seconds"] = seconds
        records.append(record)

    pcm = lead + line
    audio = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, sample_rate, 1),
        "audio/wav",
        sampleRate=sample_rate,
        encoding="wav",
        duration=seconds_of(pcm),
    )
    return audio, records


class RecordedSpeechTTS(DeepEvalBaseTTS):
    """Wraps a TTS model so lines with a recording are played, not
    synthesized.

    Everything else, streaming, cost and voice instructions, is the wrapped
    model's.
    """

    def __init__(self, inner: DeepEvalBaseTTS):
        self.inner = inner
        self.sample_rate = getattr(inner, "sample_rate", 24000)
        self.clips: Dict[str, Audio] = {}
        super().__init__(model=self.get_model_name())

    def load_model(self, *args: Any, **kwargs: Any) -> DeepEvalBaseTTS:
        return self.inner

    def synthesize(
        self, text: str, *args: Any, **kwargs: Any
    ) -> Tuple[Audio, Optional[float]]:
        if text in self.clips:
            return self.clips[text], None
        return self.inner.synthesize(text, *args, **kwargs)

    async def a_synthesize(
        self, text: str, *args: Any, **kwargs: Any
    ) -> Tuple[Audio, Optional[float]]:
        if text in self.clips:
            return self.clips[text], None
        return await self.inner.a_synthesize(text, *args, **kwargs)

    async def a_synthesize_stream(
        self, text: str, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[AudioChunk, None]:
        if text not in self.clips:
            async for chunk in self.inner.a_synthesize_stream(
                text, *args, **kwargs
            ):
                yield chunk
            return
        pcm = _pcm(self.clips[text], self.sample_rate)

        async def _whole() -> AsyncGenerator[bytes, None]:
            yield pcm

        async for chunk in frame_pcm_stream(
            _whole(), sample_rate=self.sample_rate
        ):
            yield chunk

    # A TTS model may only duck-type the base class, so what it does not say
    # about itself is read as the base class's answer.
    def _inner(self, method: str, default: Any, *args: Any) -> Any:
        answer = getattr(self.inner, method, None)
        return answer(*args) if callable(answer) else default

    def supports_streaming(self) -> bool:
        return self._inner("supports_streaming", False)

    def supports_voice_instructions(self) -> bool:
        return self._inner("supports_voice_instructions", False)

    def synthesis_cost(self, text: str) -> Optional[float]:
        if text in self.clips:
            return None
        return self._inner("synthesis_cost", None, text)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self._inner("get_model_name", type(self.inner).__name__)
