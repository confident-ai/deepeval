"""Mix a persona's background audio into the simulated caller's uplink.

The agent hears the caller as if they were in a cafe, a car, or an open-plan
office, which is what most real calls sound like.
"""

from __future__ import annotations

import logging
import os
from array import array
from functools import lru_cache
from typing import Dict, Optional, Tuple

from deepeval.dataset import BackgroundNoiseSettings
from deepeval.test_case import Audio, AudioChunk
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


class BackgroundMixer:
    """Lays a looping background bed under speech, in as many pieces as needed.

    An utterance that is streamed out as it is synthesized arrives in frames,
    and each frame has to be mixed on its own. Mixing each one independently
    would restart the bed at every frame, turning continuous ambience into the
    same fraction of a second stuttering over and over, so the mixer keeps its
    place in the loop between calls.

    One mixer covers one utterance, which leaves the result identical to mixing
    that utterance in a single pass.
    """

    def __init__(self, settings: Optional[BackgroundNoiseSettings]):
        self._settings = settings
        self._gain = settings.volume if settings is not None else 0.0
        self._beds: Dict[int, array] = {}
        self._unavailable = False
        self._offset = 0

    @property
    def enabled(self) -> bool:
        return (
            self._settings is not None
            and self._gain > 0
            and not self._unavailable
        )

    def mix_pcm(self, pcm: bytes, sample_rate: int) -> bytes:
        """Mix the next piece of speech, returning it unchanged if there is no bed."""
        if not self.enabled or not pcm:
            return pcm
        bed = self._bed_at(sample_rate)
        if bed is None or not len(bed):
            return pcm

        speech = array("h")
        speech.frombytes(pcm)
        length = len(bed)
        offset = self._offset
        gain = self._gain
        for index in range(len(speech)):
            mixed = speech[index] + int(bed[(offset + index) % length] * gain)
            speech[index] = max(_INT16_MIN, min(_INT16_MAX, mixed))
        self._offset = (offset + len(speech)) % length
        return speech.tobytes()

    def mix_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """Mix one frame of a speech stream, leaving it alone if there is no bed."""
        if not self.enabled or not chunk.sampleRate:
            return chunk
        mixed = self.mix_pcm(chunk.get_bytes(), chunk.sampleRate)
        if not self.enabled:
            return chunk
        return AudioChunk.from_bytes(
            mixed,
            chunk.mimeType,
            sampleRate=chunk.sampleRate,
            encoding=chunk.encoding,
            timestamp=chunk.timestamp,
            duration=chunk.duration,
            final=chunk.final,
        )

    def _bed_at(self, sample_rate: int) -> Optional[array]:
        bed = self._beds.get(sample_rate)
        if bed is not None:
            return bed
        try:
            bed_pcm, bed_rate = _load_pcm(self._settings.audio)
        except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
            # A missing ambience file should cost a simulation its realism, not
            # fail the run.
            logger.warning("Skipping background audio: %s", exc)
            self._unavailable = True
            return None
        bed_pcm = audio_utils.resample_pcm16(bed_pcm, bed_rate, sample_rate)
        bed = array("h")
        bed.frombytes(bed_pcm)
        self._beds[sample_rate] = bed
        return bed


def mix_background(
    audio: Audio, settings: Optional[BackgroundNoiseSettings]
) -> Audio:
    """Return `audio` with the background looped underneath it.

    Returns the input untouched when there is no background, when the volume
    is zero, or when the file cannot be read.
    """
    mixer = BackgroundMixer(settings)
    if not mixer.enabled:
        return audio

    try:
        speech_pcm, speech_rate, speech_channels = (
            audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
        )
    except ValueError as exc:
        logger.warning("Skipping background audio: %s", exc)
        return audio

    speech_pcm = audio_utils.downmix_to_mono(speech_pcm, speech_channels)
    mixed = mixer.mix_pcm(speech_pcm, speech_rate)
    if not mixer.enabled:
        return audio

    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(mixed, speech_rate, 1),
        "audio/wav",
        sampleRate=speech_rate,
        encoding="wav",
        duration=audio.duration,
        start_time=audio.start_time,
    )
