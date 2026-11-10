"""Carry the simulated caller's uplink over the line their persona calls on.

A telephone network carries only 300-3400 Hz and codes every sample as 8-bit
mu-law, so a caller on a phone sounds thin and slightly gritty. An agent behind
a phone number hears every caller that way, and testing it on studio-quality
speech overstates how well it understands them.

The line is applied after background noise is mixed in, because the noise
reaches the agent over the same channel as the voice.
"""

from __future__ import annotations

import logging
import math
import wave
from array import array
from typing import TYPE_CHECKING, List, Optional, Tuple

from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.connectors import audio_utils

if TYPE_CHECKING:
    from deepeval.dataset.golden import LineQuality

logger = logging.getLogger(__name__)

# The passband of a narrowband (G.711) telephone channel.
_PHONE_LOW_HZ = 300.0
_PHONE_HIGH_HZ = 3400.0
# Keeps the low-pass below Nyquist at sample rates too low to reach 3400 Hz.
_MAX_CUTOFF_FRACTION_OF_RATE = 0.45
_BUTTERWORTH_Q = 1 / math.sqrt(2)
_INT16_MIN = -32768
_INT16_MAX = 32767

_Coefficients = Tuple[float, float, float, float, float]


class _Biquad:
    """One second-order filter section, keeping its state between calls."""

    def __init__(self, coefficients: _Coefficients):
        self.b0, self.b1, self.b2, self.a1, self.a2 = coefficients
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, samples: List[float]) -> List[float]:
        # Locals, not attributes, in the per-sample loop: it runs on every
        # sample of every utterance, inside the live call.
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        out: List[float] = []
        append = out.append
        for x in samples:
            y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, x
            y2, y1 = y1, y
            append(y)
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return out


class _PhoneChannel:
    """A telephone channel at one sample rate: band-pass, then mu-law."""

    def __init__(self, sample_rate: int):
        cutoff = min(_PHONE_HIGH_HZ, sample_rate * _MAX_CUTOFF_FRACTION_OF_RATE)
        # The low-pass runs twice on purpose: a steeper, fourth-order cut
        # above the band.
        self._sections = [
            _Biquad(_highpass(_PHONE_LOW_HZ, sample_rate)),
            _Biquad(_lowpass(cutoff, sample_rate)),
            _Biquad(_lowpass(cutoff, sample_rate)),
        ]

    def process(self, pcm: bytes) -> bytes:
        samples: List[float] = list(array("h", pcm))
        for section in self._sections:
            samples = section.process(samples)
        band = array(
            "h",
            (max(_INT16_MIN, min(_INT16_MAX, round(s))) for s in samples),
        ).tobytes()
        return audio_utils.ulaw_to_pcm16(audio_utils.pcm16_to_ulaw(band))


class LineQualityFilter:
    """Carries one utterance over the caller's line, in as many pieces as needed.

    A streamed utterance arrives in frames. The channel's filters remember
    their last samples, so filtering frame by frame gives the same audio as
    filtering the utterance whole, with no click at each frame boundary. One
    filter covers one utterance; a wideband line leaves audio untouched.
    """

    def __init__(self, line_quality: "LineQuality"):
        self._phone = line_quality == "phone"
        self._channel: Optional[_PhoneChannel] = None
        self._sample_rate: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self._phone

    def filter_pcm(self, pcm: bytes, sample_rate: int) -> bytes:
        """Filter the next piece of mono 16-bit PCM."""
        if not self._phone or not pcm:
            return pcm
        if self._channel is None or sample_rate != self._sample_rate:
            self._channel = _PhoneChannel(sample_rate)
            self._sample_rate = sample_rate
        return self._channel.process(pcm)

    def filter_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """Filter one frame of an utterance stream."""
        if not self._phone or not chunk.sampleRate:
            return chunk
        return AudioChunk.from_bytes(
            self.filter_pcm(chunk.get_bytes(), chunk.sampleRate),
            chunk.mimeType,
            sampleRate=chunk.sampleRate,
            encoding=chunk.encoding,
            timestamp=chunk.timestamp,
            duration=chunk.duration,
            final=chunk.final,
        )


def apply_line_quality(audio: Audio, line_quality: "LineQuality") -> Audio:
    """Return `audio` as the agent hears it over the caller's line.

    Returns the input untouched on a wideband line, or when it is not 16-bit
    PCM WAV.
    """
    line = LineQualityFilter(line_quality)
    if not line.enabled:
        return audio

    try:
        pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(
            audio.get_bytes()
        )
    except (wave.Error, EOFError, ValueError) as exc:
        logger.warning("Skipping the phone line: %s", exc)
        return audio

    pcm = line.filter_pcm(
        audio_utils.downmix_to_mono(pcm, channels), sample_rate
    )
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, sample_rate, 1),
        "audio/wav",
        sampleRate=sample_rate,
        encoding="wav",
        duration=audio.duration,
        start_time=audio.start_time,
    )


def _highpass(cutoff: float, sample_rate: int) -> _Coefficients:
    cos_w, alpha = _angular(cutoff, sample_rate)
    a0 = 1 + alpha
    return (
        (1 + cos_w) / 2 / a0,
        -(1 + cos_w) / a0,
        (1 + cos_w) / 2 / a0,
        -2 * cos_w / a0,
        (1 - alpha) / a0,
    )


def _lowpass(cutoff: float, sample_rate: int) -> _Coefficients:
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
    return math.cos(w), math.sin(w) / (2 * _BUTTERWORTH_Q)
