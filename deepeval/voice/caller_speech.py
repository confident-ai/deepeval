"""Shape the simulated caller's speech the way a persona's settings ask.

A persona's speech patterns, line and background speakers all change the
caller's audio rather than their words, so they are applied where the audio is
made: a TTS wrapper that plans each utterance as speech and silence, lays any
background speaker under it, and then passes the result down the line.
"""

from __future__ import annotations

import random
import zlib
from array import array
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
)

from deepeval.models.base_model import DeepEvalBaseTTS
from deepeval.models.tts._frames import frame_pcm_stream
from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.connectors import audio_utils

if TYPE_CHECKING:
    from deepeval.dataset.golden import BackgroundSpeaker, LineSettings, Persona

# Chance that a pattern shows up in any one utterance.
PATTERN_CHANCE = {"rare": 0.15, "normal": 0.35, "frequent": 0.6}
# Mean seconds of speech between two faults on the line.
FAULT_GAP_SECONDS = {"rare": 10.0, "normal": 5.0, "frequent": 2.5}
DROPOUT_SECONDS = (0.15, 0.45)
GARBLE_SECONDS = (0.25, 0.6)
PAUSE_SECONDS = (1.5, 2.5)
SLOW_ANSWER_SECONDS = (2.0, 4.0)
ASIDE_GAP_SECONDS = 0.5
# An aside is spoken away from the phone, so it reaches the agent quieter.
ASIDE_GAIN = 0.4
STATIC_PEAK = 2500
MIN_WORDS_TO_PAUSE = 6

DEFAULT_ASIDES = (
    "Hang on, I'm on the phone.",
    "Just a second, sorry.",
    "No, not that one, the other one.",
)
DEFAULT_BACKGROUND_LINES = (
    "Are you nearly done?",
    "Where did you put it?",
    "Okay, I'll wait.",
)


@dataclass
class _Segment:
    """One stretch of an utterance: something said, or a silence."""

    text: Optional[str] = None
    silence_s: float = 0.0
    gain: float = 1.0


def split_for_pause(text: str) -> Optional[Tuple[str, str]]:
    """Where a caller would stop mid-sentence: the comma nearest the middle."""
    words = text.split()
    if len(words) < MIN_WORDS_TO_PAUSE:
        return None
    middle = len(words) // 2
    commas = [
        index + 1
        for index, word in enumerate(words[:-1])
        if word.endswith(",") and 2 <= index + 1 <= len(words) - 2
    ]
    cut = min(commas, key=lambda index: abs(index - middle), default=middle)
    return " ".join(words[:cut]), " ".join(words[cut:])


def _scaled(pcm: bytes, gain: float) -> bytes:
    if gain == 1.0 or not pcm:
        return pcm
    samples = array("h", pcm)
    for index, sample in enumerate(samples):
        samples[index] = max(-32768, min(32767, int(sample * gain)))
    return samples.tobytes()


def _silence(seconds: float, sample_rate: int) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


class _Fault:
    """One kind of damage and the next stretch of audio it falls on."""

    def __init__(
        self,
        kind: str,
        frequency: str,
        lengths: Tuple[float, float],
        sample_rate: int,
        rng: random.Random,
    ):
        self.kind = kind
        self.gap = FAULT_GAP_SECONDS[frequency]
        self.lengths = lengths
        self.sample_rate = sample_rate
        self.rng = random.Random(rng.random())
        self.start = self.stop = 0
        self.schedule(after=0)

    def schedule(self, after: int) -> None:
        wait = self.rng.expovariate(1 / self.gap)
        self.start = after + int(wait * self.sample_rate)
        self.stop = self.start + int(
            self.rng.uniform(*self.lengths) * self.sample_rate
        )


class LineFaults:
    """Dropouts, garbled spans and static on 16-bit mono PCM.

    Keeps its place between calls, so an utterance fed frame by frame is
    damaged exactly as it would be fed whole.
    """

    def __init__(
        self, sample_rate: int, line: "LineSettings", rng: random.Random
    ):
        self.sample_rate = sample_rate
        self.static = line.static
        self.position = 0
        # Each kind of damage draws from its own sequence, so where one falls
        # never depends on how the audio was cut into frames.
        self.static_rng = random.Random(rng.random())
        self.faults: List[_Fault] = []
        if line.dropouts is not None:
            self.faults.append(
                _Fault(
                    "dropout", line.dropouts, DROPOUT_SECONDS, sample_rate, rng
                )
            )
        if line.garbled is not None:
            self.faults.append(
                _Fault("garble", line.garbled, GARBLE_SECONDS, sample_rate, rng)
            )

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        samples = array("h", pcm)
        begin, end = self.position, self.position + len(samples)
        for fault in self.faults:
            while fault.start < end:
                self._damage(
                    samples,
                    fault.kind,
                    max(fault.start, begin) - begin,
                    min(fault.stop, end) - begin,
                )
                if fault.stop > end:
                    break
                fault.schedule(after=fault.stop)
        if self.static > 0:
            peak = int(self.static * STATIC_PEAK)
            for index, sample in enumerate(samples):
                noisy = sample + self.static_rng.randint(-peak, peak)
                samples[index] = max(-32768, min(32767, noisy))
        self.position = end
        return samples.tobytes()

    @staticmethod
    def _damage(samples: array, kind: str, start: int, stop: int) -> None:
        if kind == "dropout":
            for index in range(start, stop):
                samples[index] = 0
            return
        # Sample-and-hold at a coarse bit depth: the robotic smear of a
        # connection that is losing packets without dropping the call.
        held = 0
        for index in range(start, stop):
            if (index - start) % 10 == 0:
                held = (samples[index] // 4096) * 4096
            samples[index] = held


class _Overlay:
    """A background speaker's line, mixed under the caller as audio passes."""

    def __init__(self, pcm: bytes, start: int):
        self.samples = array("h", pcm)
        self.start = start
        self.position = 0

    def mix(self, pcm: bytes) -> bytes:
        speech = array("h", pcm)
        begin = self.position
        self.position += len(speech)
        first = max(self.start, begin)
        last = min(self.start + len(self.samples), self.position)
        for index in range(first, last):
            mixed = speech[index - begin] + self.samples[index - self.start]
            speech[index - begin] = max(-32768, min(32767, mixed))
        return speech.tobytes()

    def remainder(self) -> bytes:
        """What the speaker was still saying when the caller stopped."""
        spoken = max(0, self.position - self.start)
        return self.samples[spoken:].tobytes()


class CallerSpeechTTS(DeepEvalBaseTTS):
    """Wraps a TTS model so the caller talks the way their persona does.

    Streaming, cost and voice instructions are all the wrapped model's.
    """

    def __init__(self, inner: DeepEvalBaseTTS, persona: "Persona"):
        self.inner = inner
        self.sample_rate = inner.sample_rate
        self.patterns = persona.speech_patterns
        self.line = persona.line
        self.volume = persona.voice.volume if persona.voice is not None else 1.0
        noise = persona.background_noise
        self.speakers: List["BackgroundSpeaker"] = (
            list(noise.other_speakers) if noise is not None else []
        )
        # Seeded from the persona, so the same caller makes the same call.
        self.rng = random.Random(
            zlib.crc32(f"{persona.name}|{persona.characteristics}".encode())
        )
        self.asides: List[str] = list(DEFAULT_ASIDES)
        self.background_lines: List[List[str]] = [
            list(DEFAULT_BACKGROUND_LINES) for _ in self.speakers
        ]
        self._clips: Dict[Tuple[int, str], bytes] = {}
        self.extra_cost = 0.0
        super().__init__(model=inner.get_model_name())

    @staticmethod
    def applies_to(persona: "Persona") -> bool:
        patterns, line, noise = (
            persona.speech_patterns,
            persona.line,
            persona.background_noise,
        )
        return bool(
            (persona.voice is not None and persona.voice.volume != 1.0)
            or (patterns is not None and patterns.shapes_audio)
            or (
                line is not None
                and (
                    line.dropouts
                    or line.garbled
                    or line.static
                    or line.delay_ms
                )
            )
            or (noise is not None and noise.other_speakers)
        )

    def load_model(self, *args: Any, **kwargs: Any) -> DeepEvalBaseTTS:
        return self.inner

    def use_lines(
        self,
        asides: Optional[List[str]] = None,
        background_lines: Optional[List[List[str]]] = None,
    ) -> None:
        """Replace the stock asides and background lines with written ones."""
        if asides:
            self.asides = list(asides)
        for index, lines in enumerate(background_lines or []):
            if lines and index < len(self.background_lines):
                self.background_lines[index] = list(lines)

    def _happens(self, frequency: Optional[str]) -> bool:
        return (
            frequency is not None
            and self.rng.random() < PATTERN_CHANCE[frequency]
        )

    def plan(self, text: str) -> List[_Segment]:
        """One utterance as the stretches of speech and silence it is made of."""
        segments: List[_Segment] = []
        if self.line is not None and self.line.delay_ms:
            segments.append(_Segment(silence_s=self.line.delay_ms / 1000))
        patterns = self.patterns
        if patterns is None:
            return segments + [_Segment(text=text)]
        if self._happens(patterns.slow_to_answer):
            segments.append(
                _Segment(silence_s=self.rng.uniform(*SLOW_ANSWER_SECONDS))
            )
        if self._happens(patterns.side_speech):
            segments.append(
                _Segment(text=self.rng.choice(self.asides), gain=ASIDE_GAIN)
            )
            segments.append(_Segment(silence_s=ASIDE_GAP_SECONDS))
        halves = (
            split_for_pause(text)
            if self._happens(patterns.mid_sentence_pauses)
            else None
        )
        if halves is None:
            return segments + [_Segment(text=text)]
        return segments + [
            _Segment(text=halves[0]),
            _Segment(silence_s=self.rng.uniform(*PAUSE_SECONDS)),
            _Segment(text=halves[1]),
        ]

    def _pick_background_line(self) -> Optional[Tuple[int, str]]:
        for index, speaker in enumerate(self.speakers):
            if self._happens(speaker.frequency):
                return index, self.rng.choice(self.background_lines[index])
        return None

    def _speaker_kwargs(self, index: int) -> Dict:
        voice = self.speakers[index].voice
        return {} if voice is None or voice.id is None else {"voice": voice.id}

    def _pcm(self, audio: Audio, gain: float, sample_rate: int) -> bytes:
        """One synthesized clip as mono PCM at the utterance's rate and gain."""
        pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
        pcm = audio_utils.downmix_to_mono(pcm, channels)
        if rate != sample_rate:
            pcm = audio_utils.resample_pcm16(pcm, rate, sample_rate)
        return _scaled(pcm, gain)

    def _keep_clip(
        self, picked: Tuple[int, str], spoken: Tuple[Audio, Optional[float]]
    ) -> None:
        """Remember a background speaker's line, so it is paid for once."""
        audio, cost = spoken
        self.extra_cost += cost or 0.0
        self._clips[picked] = self._pcm(
            audio, self.speakers[picked[0]].volume, self.sample_rate
        )

    def _overlay(self, picked: Optional[Tuple[int, str]]) -> Optional[_Overlay]:
        if picked is None:
            return None
        start = int(self.rng.uniform(0.2, 1.0) * self.sample_rate)
        return _Overlay(self._clips[picked], start)

    def _faults(self) -> Optional[LineFaults]:
        line = self.line
        if line is None or not (line.dropouts or line.garbled or line.static):
            return None
        return LineFaults(self.sample_rate, line, self.rng)

    def _assemble(
        self,
        segments: List[_Segment],
        spoken: List[Tuple[Audio, Optional[float]]],
        picked: Optional[Tuple[int, str]],
    ) -> Tuple[Audio, Optional[float]]:
        """Join what was synthesized for each segment into one utterance."""
        clips = iter(spoken)
        pcm = b"".join(
            (
                _silence(segment.silence_s, self.sample_rate)
                if segment.text is None
                else self._pcm(
                    next(clips)[0], segment.gain * self.volume, self.sample_rate
                )
            )
            for segment in segments
        )
        overlay = self._overlay(picked)
        if overlay is not None:
            pcm = overlay.mix(pcm) + overlay.remainder()
        faults = self._faults()
        if faults is not None:
            pcm = faults.process(pcm)
        costs = [cost for _, cost in spoken if cost is not None]
        audio = Audio.from_bytes(
            audio_utils.pcm16_to_wav_bytes(pcm, self.sample_rate, 1),
            "audio/wav",
            sampleRate=self.sample_rate,
            encoding="wav",
            duration=len(pcm) / 2 / self.sample_rate,
        )
        return audio, self._with_extra(sum(costs) if costs else None)

    def synthesize(
        self, text: str, *args: Any, voice: Optional[str] = None, **kwargs: Any
    ) -> Tuple[Audio, Optional[float]]:
        segments = self.plan(text)
        spoken = [
            self.inner.synthesize(segment.text, *args, voice=voice, **kwargs)
            for segment in segments
            if segment.text is not None
        ]
        picked = self._pick_background_line()
        if picked is not None and picked not in self._clips:
            self._keep_clip(
                picked,
                self.inner.synthesize(
                    picked[1], **self._speaker_kwargs(picked[0])
                ),
            )
        return self._assemble(segments, spoken, picked)

    async def a_synthesize(
        self, text: str, *args: Any, voice: Optional[str] = None, **kwargs: Any
    ) -> Tuple[Audio, Optional[float]]:
        segments = self.plan(text)
        spoken = [
            await self.inner.a_synthesize(
                segment.text, *args, voice=voice, **kwargs
            )
            for segment in segments
            if segment.text is not None
        ]
        picked = await self._a_background_line()
        return self._assemble(segments, spoken, picked)

    async def _a_background_line(self) -> Optional[Tuple[int, str]]:
        picked = self._pick_background_line()
        if picked is not None and picked not in self._clips:
            self._keep_clip(
                picked,
                await self.inner.a_synthesize(
                    picked[1], **self._speaker_kwargs(picked[0])
                ),
            )
        return picked

    async def a_synthesize_stream(
        self, text: str, *args: Any, voice: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[AudioChunk, None]:
        sample_rate = self.sample_rate
        segments = self.plan(text)
        overlay = self._overlay(await self._a_background_line())
        faults = self._faults()
        for segment in segments:
            if segment.text is not None and segment.gain != 1.0:
                self.extra_cost += (
                    self.inner.synthesis_cost(segment.text) or 0.0
                )

        def _finish(pcm: bytes) -> bytes:
            if overlay is not None:
                pcm = overlay.mix(pcm)
            return faults.process(pcm) if faults is not None else pcm

        async def _pcm() -> AsyncGenerator[bytes, None]:
            for segment in segments:
                if segment.text is None:
                    yield _finish(_silence(segment.silence_s, sample_rate))
                    continue
                async for chunk in self.inner.a_synthesize_stream(
                    segment.text, *args, voice=voice, **kwargs
                ):
                    pcm = chunk.get_bytes()
                    if chunk.sampleRate and chunk.sampleRate != sample_rate:
                        pcm = audio_utils.resample_pcm16(
                            pcm, chunk.sampleRate, sample_rate
                        )
                    yield _finish(_scaled(pcm, segment.gain * self.volume))
            if overlay is not None:
                rest = overlay.remainder()
                yield faults.process(rest) if faults is not None else rest

        async for chunk in frame_pcm_stream(_pcm(), sample_rate=sample_rate):
            yield chunk

    def _with_extra(self, cost: Optional[float]) -> Optional[float]:
        """Fold in what the background speakers cost since the last utterance."""
        extra, self.extra_cost = self.extra_cost, 0.0
        if cost is None and not extra:
            return None
        return (cost or 0.0) + extra

    def supports_streaming(self) -> bool:
        return self.inner.supports_streaming()

    def supports_voice_instructions(self) -> bool:
        return self.inner.supports_voice_instructions()

    def synthesis_cost(self, text: str) -> Optional[float]:
        return self._with_extra(self.inner.synthesis_cost(text))

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self.inner.get_model_name()
