"""Mix a persona's background audio into the simulated caller's uplink.

The agent hears the caller as if they were in a cafe, a car, or an open-plan
office, which is what most real calls sound like.
"""

from __future__ import annotations

import logging
import os
from array import array
from functools import lru_cache
from typing import Optional, Tuple

from deepeval.dataset import BackgroundNoiseSettings
from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils

logger = logging.getLogger(__name__)

_INT16_MIN = -32768
_INT16_MAX = 32767


@lru_cache(maxsize=8)
def _load_pcm(path: str) -> Tuple[bytes, int]:
    """Decode a background file to mono 16-bit PCM, cached across turns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Background audio file not found: {path}")
    if path.lower().endswith(".wav"):
        with open(path, "rb") as handle:
            wav_bytes = handle.read()
    else:
        wav_bytes = _decode_to_wav(path)
    pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(wav_bytes)
    return audio_utils.downmix_to_mono(pcm, channels), sample_rate


def _decode_to_wav(path: str) -> bytes:
    try:
        from pydub import AudioSegment
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Background audio in a non-WAV format needs pydub and ffmpeg "
            "('pip install pydub'), or supply a .wav file instead."
        ) from exc

    buffer = AudioSegment.from_file(path).export(format="wav")
    try:
        return buffer.read()
    finally:
        buffer.close()


def _loop_to_length(pcm: bytes, length: int) -> bytes:
    if not pcm:
        return b"\x00" * length
    repeats = -(-length // len(pcm))  # ceil
    return (pcm * repeats)[:length]


def mix_background(
    audio: Audio, settings: Optional[BackgroundNoiseSettings]
) -> Audio:
    """Return `audio` with the background looped underneath it.

    Returns the input untouched when there is no background, when the volume
    is zero, or when the file cannot be read — a missing ambience file should
    degrade the realism of a simulation, not fail the run.
    """
    if settings is None or settings.volume <= 0:
        return audio

    try:
        bed_pcm, bed_rate = _load_pcm(settings.audio)
        speech_pcm, speech_rate, speech_channels = (
            audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
        )
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        logger.warning("Skipping background audio: %s", exc)
        return audio

    speech_pcm = audio_utils.downmix_to_mono(speech_pcm, speech_channels)
    bed_pcm = audio_utils.resample_pcm16(bed_pcm, bed_rate, speech_rate)
    bed_pcm = _loop_to_length(bed_pcm, len(speech_pcm))

    speech = array("h")
    speech.frombytes(speech_pcm)
    bed = array("h")
    bed.frombytes(bed_pcm)

    gain = settings.volume
    for index in range(len(speech)):
        mixed = speech[index] + int(bed[index] * gain)
        speech[index] = max(_INT16_MIN, min(_INT16_MAX, mixed))

    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(speech.tobytes(), speech_rate, 1),
        "audio/wav",
        sampleRate=speech_rate,
        encoding="wav",
        duration=audio.duration,
        start_time=audio.start_time,
    )
