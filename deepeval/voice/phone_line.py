import math
from array import array
from typing import Any, AsyncGenerator, List, Optional, Tuple

from deepeval.models.base_model import DeepEvalBaseTTS
from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.connectors import audio_utils

# The passband of a narrowband (G.711) telephone channel.
PHONE_LOW_HZ = 300
PHONE_HIGH_HZ = 3400
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


class PhoneLine:
    """Makes 16-bit mono PCM sound like it crossed a telephone line.

    Band-limits to 300–3400 Hz and round-trips through μ-law. The filter
    state carries across calls, so feeding a signal frame by frame gives
    exactly the bytes of feeding it whole.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        cutoff = min(PHONE_HIGH_HZ, sample_rate * 0.45)
        self.filters = [
            Biquad(_highpass(PHONE_LOW_HZ, sample_rate)),
            Biquad(_lowpass(cutoff, sample_rate)),
            Biquad(_lowpass(cutoff, sample_rate)),
        ]

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        samples: List[float] = list(array("h", pcm))
        for biquad in self.filters:
            samples = biquad.process(samples)
        band = array(
            "h", (max(-32768, min(32767, round(sample))) for sample in samples)
        ).tobytes()
        return audio_utils.ulaw_to_pcm16(audio_utils.pcm16_to_ulaw(band))


class PhoneLineTTS(DeepEvalBaseTTS):
    """Wraps a TTS model so everything it says goes through a `PhoneLine`.

    Streaming, cost and voice instructions are all the wrapped model's.
    """

    def __init__(self, inner: DeepEvalBaseTTS):
        self.inner = inner
        self.sample_rate = inner.sample_rate
        super().__init__(model=inner.get_model_name())

    def load_model(self, *args: Any, **kwargs: Any) -> DeepEvalBaseTTS:
        return self.inner

    def synthesize(
        self, text: str, *args: Any, voice: Optional[str] = None, **kwargs: Any
    ) -> Tuple[Audio, Optional[float]]:
        audio, cost = self.inner.synthesize(text, *args, voice=voice, **kwargs)
        return phone_line_audio(audio), cost

    async def a_synthesize(
        self, text: str, *args: Any, voice: Optional[str] = None, **kwargs: Any
    ) -> Tuple[Audio, Optional[float]]:
        audio, cost = await self.inner.a_synthesize(
            text, *args, voice=voice, **kwargs
        )
        return phone_line_audio(audio), cost

    async def a_synthesize_stream(
        self, text: str, *args: Any, voice: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[AudioChunk, None]:
        line: Optional[PhoneLine] = None
        async for chunk in self.inner.a_synthesize_stream(
            text, *args, voice=voice, **kwargs
        ):
            line = line or PhoneLine(chunk.sampleRate or self.sample_rate)
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


def phone_line_audio(audio: Audio) -> Audio:
    pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(
        audio.get_bytes()
    )
    pcm = PhoneLine(sample_rate).process(
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
