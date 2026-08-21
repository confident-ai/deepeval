"""Dependency-free acoustic measurements used by voice metrics."""

from __future__ import annotations

import hashlib
import math
from array import array
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import List, Optional, Sequence, Tuple

from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils


@dataclass
class AudioMeasurements:
    duration: float
    rms_dbfs: float
    clipping_fraction: float
    silence_fraction: float
    longest_silence_ms: float
    dropout_events: int
    loop_events: int
    estimated_snr_db: float
    zero_crossing_rate: float
    pitch_mean_hz: Optional[float]
    pitch_variation_hz: Optional[float]
    starts_abruptly: bool
    ends_abruptly: bool


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def decode_audio(audio: Audio) -> Tuple[array, int]:
    try:
        pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(
            audio.get_bytes()
        )
    except Exception as error:
        raise ValueError("Audio is not a decodable PCM WAV.") from error
    pcm = audio_utils.downmix_to_mono(pcm, channels or 1)
    samples = array("h")
    samples.frombytes(pcm)
    if not sample_rate or not samples:
        raise ValueError("Audio contains no decodable PCM samples.")
    audio.duration = len(samples) / sample_rate
    return samples, sample_rate


def _rms(samples: Sequence[int]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _dbfs(rms: float) -> float:
    if rms <= 0:
        return -96.0
    return 20.0 * math.log10(rms / 32768.0)


def _frame_samples(
    samples: Sequence[int], sample_rate: int, frame_ms: int = 20
) -> List[Sequence[int]]:
    width = max(1, int(sample_rate * frame_ms / 1000))
    return [
        samples[index : index + width]
        for index in range(0, len(samples), width)
        if len(samples[index : index + width]) >= width // 2
    ]


def _pitch_hz(frame: Sequence[int], sample_rate: int) -> Optional[float]:
    if _rms(frame) < audio_utils.DEFAULT_SILENCE_RMS:
        return None
    step = max(1, sample_rate // 8000)
    data = frame[::step]
    rate = sample_rate / step
    min_lag = max(1, int(rate / 400))
    max_lag = min(len(data) // 2, int(rate / 70))
    if max_lag <= min_lag:
        return None
    centered_mean = sum(data) / len(data)
    centered = [sample - centered_mean for sample in data]
    energy = sum(sample * sample for sample in centered)
    if energy <= 0:
        return None
    best_lag = min_lag
    best_correlation = float("-inf")
    for lag in range(min_lag, max_lag + 1):
        correlation = sum(
            centered[index] * centered[index + lag]
            for index in range(len(centered) - lag)
        )
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    if best_correlation / energy < 0.15:
        return None
    return rate / best_lag


def analyze_audio(audio: Audio) -> AudioMeasurements:
    samples, sample_rate = decode_audio(audio)
    frames = _frame_samples(samples, sample_rate)
    frame_rms = [_rms(frame) for frame in frames]
    silent = [value < audio_utils.DEFAULT_SILENCE_RMS for value in frame_rms]

    longest_silence = current_silence = 0
    for is_silent in silent:
        if is_silent:
            current_silence += 1
            longest_silence = max(longest_silence, current_silence)
        else:
            current_silence = 0

    dropout_events = 0
    index = 1
    while index < len(silent) - 1:
        if not silent[index]:
            index += 1
            continue
        start = index
        while index < len(silent) and silent[index]:
            index += 1
        length = index - start
        if (
            start > 0
            and index < len(silent)
            and not silent[start - 1]
            and not silent[index]
            and length <= 10
        ):
            dropout_events += 1

    voiced_rms = [
        value
        for value in frame_rms
        if value > 0 and value >= audio_utils.DEFAULT_SILENCE_RMS
    ]
    quiet_rms = [
        value
        for value in frame_rms
        if value > 0 and value < audio_utils.DEFAULT_SILENCE_RMS
    ]
    speech_level = (
        sorted(voiced_rms)[int(0.75 * (len(voiced_rms) - 1))]
        if voiced_rms
        else 0.0
    )
    noise_level = (
        sorted(quiet_rms)[int(0.25 * (len(quiet_rms) - 1))]
        if quiet_rms
        else max(1.0, speech_level / 100.0)
    )
    snr_db = (
        20.0 * math.log10(max(speech_level, 1.0) / max(noise_level, 1.0))
        if speech_level
        else 0.0
    )

    crossings = sum(
        1
        for previous, current in zip(samples, samples[1:])
        if (previous < 0 <= current) or (previous >= 0 > current)
    )
    pitches = [
        pitch
        for frame in _frame_samples(samples, sample_rate, frame_ms=40)
        if (pitch := _pitch_hz(frame, sample_rate)) is not None
    ]

    loop_window = max(1, int(sample_rate * 0.25))
    fingerprints = {}
    for start in range(0, len(samples) - loop_window + 1, loop_window):
        window = samples[start : start + loop_window]
        if _rms(window) < audio_utils.DEFAULT_SILENCE_RMS:
            continue
        fingerprint = hashlib.blake2b(
            array("h", window).tobytes(), digest_size=8
        ).digest()
        fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
    loop_events = sum(count - 1 for count in fingerprints.values() if count > 2)

    edge_size = max(1, int(sample_rate * 0.02))
    edge_threshold = max(audio_utils.DEFAULT_SILENCE_RMS * 2, 1000)
    return AudioMeasurements(
        duration=len(samples) / sample_rate,
        rms_dbfs=_dbfs(_rms(samples)),
        clipping_fraction=sum(abs(sample) >= 32760 for sample in samples)
        / len(samples),
        silence_fraction=sum(silent) / len(silent) if silent else 1.0,
        longest_silence_ms=float(longest_silence * 20),
        dropout_events=dropout_events,
        loop_events=loop_events,
        estimated_snr_db=snr_db,
        zero_crossing_rate=crossings / max(1, len(samples) - 1),
        pitch_mean_hz=mean(pitches) if pitches else None,
        pitch_variation_hz=pstdev(pitches) if len(pitches) > 1 else None,
        starts_abruptly=_rms(samples[:edge_size]) > edge_threshold,
        ends_abruptly=_rms(samples[-edge_size:]) > edge_threshold,
    )


def speaking_rate_wpm(
    content: str, measurements: AudioMeasurements
) -> Optional[float]:
    words = len(content.split())
    speech_duration = measurements.duration * (
        1.0 - measurements.silence_fraction
    )
    if words == 0 or speech_duration <= 0:
        return None
    return words / (speech_duration / 60.0)


def naturalness_score(
    measurements: AudioMeasurements, *, words_per_minute: Optional[float]
) -> float:
    score = 1.0
    score -= min(0.35, measurements.clipping_fraction * 25.0)
    score -= min(0.25, measurements.dropout_events * 0.08)
    score -= min(0.2, measurements.loop_events * 0.1)
    if measurements.silence_fraction > 0.45:
        score -= min(0.2, (measurements.silence_fraction - 0.45) * 0.5)
    if measurements.estimated_snr_db < 15:
        score -= min(0.2, (15 - measurements.estimated_snr_db) / 50)
    if words_per_minute is not None:
        if words_per_minute < 80:
            score -= min(0.2, (80 - words_per_minute) / 300)
        elif words_per_minute > 240:
            score -= min(0.2, (words_per_minute - 240) / 500)
    if measurements.pitch_variation_hz is not None:
        if measurements.pitch_variation_hz < 4:
            score -= 0.1
        elif measurements.pitch_variation_hz > 90:
            score -= 0.1
    return clamp_score(score)


def intelligibility_score(measurements: AudioMeasurements) -> float:
    snr = clamp_score((measurements.estimated_snr_db - 3.0) / 22.0)
    volume = clamp_score(1.0 - abs(measurements.rms_dbfs + 20.0) / 35.0)
    clipping = clamp_score(1.0 - measurements.clipping_fraction * 30.0)
    dropouts = clamp_score(1.0 - measurements.dropout_events * 0.12)
    return clamp_score(
        0.4 * snr + 0.25 * volume + 0.2 * clipping + 0.15 * dropouts
    )
