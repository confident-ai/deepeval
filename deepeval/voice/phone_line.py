import math
from array import array
from typing import Any, AsyncGenerator, List, Optional, Tuple

from deepeval.models.base_model import DeepEvalBaseTTS
from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.connectors import audio_utils

# The passband of a narrowband (G.711) telephone channel.
PHONE_LOW_HZ = 300
PHONE_HIGH_HZ = 3400
# A weak mobile connection: a narrower band, encoded in coarser steps.
MOBILE_HIGH_HZ = 2400
MOBILE_STEP = 1024
# A caller across the room: quieter, with two walls answering back.
SPEAKERPHONE_GAIN = 0.55
SPEAKERPHONE_TAPS = ((0.037, 0.45), (0.061, 0.35))
BUTTERWORTH_Q = 1 / math.sqrt(2)


class Biquad:
    """A second-order IIR filter that keeps its state between calls."""

    def __init__(self, coefficients: Tuple[float, float, float, float, float]):
        self.b0, self.b1, self.b2, self.a1, self.a2 = coefficients
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, samples: List[float]) -> List[float]:
        out = []
        for x in samples:
            y = (
                self.b0 * x
                + self.b1 * self.x1
                + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2
            )
            self.x2, self.x1, self.y2, self.y1 = self.x1, x, self.y1, y
            out.append(y)
        return out


class Echoes:
    """The reflections of a room, kept between calls like a filter's state.

    Each tap is one wall: the sound comes back after `delay` seconds at
    `gain` of its level, and that reflection is reflected again.
    """

    def __init__(self, sample_rate: int, taps: Tuple[Tuple[float, float], ...]):
        self.taps = [
            ([0.0] * max(1, int(delay * sample_rate)), gain)
            for delay, gain in taps
        ]
        self.position = 0

    def process(self, samples: List[float]) -> List[float]:
        out = []
        for sample in samples:
            wet = sample
            for line, gain in self.taps:
                index = self.position % len(line)
                wet += line[index] * gain
                line[index] = sample + line[index] * gain
            self.position += 1
            out.append(wet)
        return out


class PhoneLine:
    """Makes 16-bit mono PCM sound like it crossed a telephone line.

    "phone" band-limits to 300–3400 Hz and round-trips through μ-law.
    "mobile" narrows the band further and coarsens the encoding, the way a
    weak mobile connection does. "speakerphone" puts the caller across a room
    from the phone: quieter, with the room's reflections, on a phone line.

    The filter state carries across calls, so feeding a signal frame by frame
    gives exactly the bytes of feeding it whole.
    """

    def __init__(self, sample_rate: int, quality: str = "phone"):
        self.sample_rate = sample_rate
        self.quality = quality
        high = MOBILE_HIGH_HZ if quality == "mobile" else PHONE_HIGH_HZ
        cutoff = min(high, sample_rate * 0.45)
        self.filters = [
            Biquad(_highpass(PHONE_LOW_HZ, sample_rate)),
            Biquad(_lowpass(cutoff, sample_rate)),
            Biquad(_lowpass(cutoff, sample_rate)),
        ]
        self.room = (
            Echoes(sample_rate, SPEAKERPHONE_TAPS)
            if quality == "speakerphone"
            else None
        )

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        samples: List[float] = list(array("h", pcm))
        if self.room is not None:
            samples = [
                sample * SPEAKERPHONE_GAIN
                for sample in self.room.process(samples)
            ]
        for biquad in self.filters:
            samples = biquad.process(samples)
        step = MOBILE_STEP if self.quality == "mobile" else 1
        band = array(
            "h",
            (
                max(-32768, min(32767, round(sample / step) * step))
                for sample in samples
            ),
        ).tobytes()
        return audio_utils.ulaw_to_pcm16(audio_utils.pcm16_to_ulaw(band))


class PhoneLineTTS(DeepEvalBaseTTS):
    """Wraps a TTS model so everything it says goes through a `PhoneLine`.

    Streaming, cost and voice instructions are all the wrapped model's.
    """

    def __init__(self, inner: DeepEvalBaseTTS, quality: str = "phone"):
        self.inner = inner
        self.quality = quality
        self.sample_rate = inner.sample_rate
        super().__init__(model=inner.get_model_name())

    def load_model(self, *args: Any, **kwargs: Any) -> DeepEvalBaseTTS:
        return self.inner

    def synthesize(
        self, text: str, *args: Any, voice: Optional[str] = None, **kwargs: Any
    ) -> Tuple[Audio, Optional[float]]:
        audio, cost = self.inner.synthesize(text, *args, voice=voice, **kwargs)
        return phone_line_audio(audio, self.quality), cost

    async def a_synthesize(
        self, text: str, *args: Any, voice: Optional[str] = None, **kwargs: Any
    ) -> Tuple[Audio, Optional[float]]:
        audio, cost = await self.inner.a_synthesize(
            text, *args, voice=voice, **kwargs
        )
        return phone_line_audio(audio, self.quality), cost

    async def a_synthesize_stream(
        self, text: str, *args: Any, voice: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[AudioChunk, None]:
        line: Optional[PhoneLine] = None
        async for chunk in self.inner.a_synthesize_stream(
            text, *args, voice=voice, **kwargs
        ):
            line = line or PhoneLine(
                chunk.sampleRate or self.sample_rate, self.quality
            )
            yield AudioChunk.from_bytes(
                line.process(chunk.get_bytes()),
                chunk.mimeType,
                sampleRate=chunk.sampleRate,
                encoding=chunk.encoding,
                timestamp=chunk.timestamp,
                duration=chunk.duration,
                final=chunk.final,
            )

    def supports_streaming(self) -> bool:
        return self.inner.supports_streaming()

    def supports_voice_instructions(self) -> bool:
        return self.inner.supports_voice_instructions()

    def synthesis_cost(self, text: str) -> Optional[float]:
        return self.inner.synthesis_cost(text)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self.inner.get_model_name()


def phone_line_audio(audio: Audio, quality: str = "phone") -> Audio:
    pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(
        audio.get_bytes()
    )
    pcm = PhoneLine(sample_rate, quality).process(
        audio_utils.downmix_to_mono(pcm, channels)
    )
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, sample_rate, 1),
        "audio/wav",
        sampleRate=sample_rate,
        encoding="wav",
        duration=audio.duration,
        start_time=audio.start_time,
    )


def _highpass(
    cutoff: float, sample_rate: int
) -> Tuple[float, float, float, float, float]:
    cos_w, alpha = _angular(cutoff, sample_rate)
    a0 = 1 + alpha
    return (
        (1 + cos_w) / 2 / a0,
        -(1 + cos_w) / a0,
        (1 + cos_w) / 2 / a0,
        -2 * cos_w / a0,
        (1 - alpha) / a0,
    )


def _lowpass(
    cutoff: float, sample_rate: int
) -> Tuple[float, float, float, float, float]:
    cos_w, alpha = _angular(cutoff, sample_rate)
    a0 = 1 + alpha
    return (
        (1 - cos_w) / 2 / a0,
        (1 - cos_w) / a0,
        (1 - cos_w) / 2 / a0,
        -2 * cos_w / a0,
        (1 - alpha) / a0,
    )


def _angular(cutoff: float, sample_rate: int) -> Tuple[float, float]:
    w = 2 * math.pi * cutoff / sample_rate
    return math.cos(w), math.sin(w) / (2 * BUTTERWORTH_Q)
