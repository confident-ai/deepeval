"""Synthetic audio for the audio-glitch and audio-defect metric tests."""

import math
import random
from array import array

from deepeval.test_case import Audio, Turn
from deepeval.voice.connectors import audio_utils

RATE = 8000


def speech_like(seconds, *, base_hz=200.0, seed=0, noise=20) -> array:
    """A voice-shaped signal: pitch gliding ±30 Hz, harmonics, syllables of
    sound and near-silence, and faint room noise so pauses are never zero."""
    rng = random.Random(seed)
    out = array("h")
    phase = 0.0
    for index in range(int(seconds * RATE)):
        t = index / RATE
        hz = base_hz + 30.0 * math.sin(2 * math.pi * t / 1.0)
        phase += 2 * math.pi * hz / RATE
        envelope = max(0.0, math.sin(math.pi * 4.0 * t)) ** 0.7
        voice = (
            math.sin(phase)
            + 0.5 * math.sin(2 * phase)
            + 0.3 * math.sin(3 * phase)
        ) / 1.8
        out.append(int(8000 * envelope * voice + rng.gauss(0, noise)))
    return out


def vowel(seconds, *, start_hz=200.0, end_hz=185.0) -> array:
    """A held vowel: harmonics, steady level, and the slow pitch glide every
    voice has."""
    out = array("h")
    phase = 0.0
    count = int(seconds * RATE)
    for index in range(count):
        hz = start_hz + (end_hz - start_hz) * index / count
        phase += 2 * math.pi * hz / RATE
        out.append(
            int(
                5000
                * (
                    math.sin(phase)
                    + 0.5 * math.sin(2 * phase)
                    + 0.3 * math.sin(3 * phase)
                )
                / 1.8
            )
        )
    return out


def tone(seconds, hz, *, amplitude=6000) -> array:
    out = array("h")
    count = int(seconds * RATE)
    fade = int(0.02 * RATE)
    for index in range(count):
        envelope = max(0.0, min(1.0, index / fade, (count - index - 1) / fade))
        out.append(
            int(
                amplitude * envelope * math.sin(2 * math.pi * hz * index / RATE)
            )
        )
    return out


def spoken_words(count=8, *, pause=0.1, noise=20) -> array:
    rng = random.Random(count)
    out = array("h")
    for index in range(count):
        out.extend(speech_like(0.25, seed=index, noise=noise))
        out.extend(int(rng.gauss(0, noise)) for _ in range(int(pause * RATE)))
    return out


def synthetic_syllable(seconds, *, hz=230.0, wobble=0.005, seed=0) -> array:
    """No glide, but the pitch moves a little every 40 ms frame."""
    rng = random.Random(seed)
    out = array("h")
    phase = 0.0
    step = int(0.04 * RATE)
    for index in range(int(seconds * RATE)):
        if index % step == 0:
            frame_hz = hz * (1 + rng.gauss(0, wobble))
        phase += 2 * math.pi * frame_hz / RATE
        out.append(
            int(
                5000
                * (
                    math.sin(phase)
                    + 0.5 * math.sin(2 * phase)
                    + 0.3 * math.sin(3 * phase)
                )
                / 1.8
            )
        )
    return out


def zeros(seconds) -> array:
    return array("h", [0] * int(seconds * RATE))


def clipped(seconds) -> array:
    return array(
        "h",
        [
            max(-32768, min(32767, sample * 8))
            for sample in speech_like(seconds)
        ],
    )


def cut_short(samples: array, seconds) -> array:
    return samples[: int(seconds * RATE)]


def _saturate(value) -> int:
    return max(-32768, min(32767, int(value)))


def with_noise(samples: array, dbfs, *, seed=0) -> array:
    rng = random.Random(seed)
    level = 32768 * 10 ** (dbfs / 20)
    return array("h", [_saturate(v + rng.gauss(0, level)) for v in samples])


def to_audio(samples: array, start_time=None) -> Audio:
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(samples.tobytes(), RATE, num_channels=1),
        "audio/wav",
        sampleRate=RATE,
        encoding="wav",
        duration=len(samples) / RATE,
        start_time=start_time,
    )


def clean_turns(count=6):
    turns = []
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        turns.append(
            Turn(
                role=role,
                content=f"Turn {index} spoken by the {role}.",
                audio=to_audio(
                    speech_like(
                        3.0,
                        base_hz=200.0 if role == "user" else 230.0,
                        seed=index,
                    ),
                    start_time=index * 3.5,
                ),
            )
        )
    return turns
