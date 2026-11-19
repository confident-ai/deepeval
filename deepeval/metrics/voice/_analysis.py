"""Acoustic measurements used by voice metrics. They need numpy."""

from __future__ import annotations

import hashlib
import math
from array import array
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Any, List, Optional, Sequence, Tuple

from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils

DROPOUT_MAX_FRAMES = 10
DROPOUT_FLOOR_RATIO = 0.1
MIN_PAUSE_FLOOR_RMS = 8.0
# With no pause noise to compare against, a dropout is a near-zero run whose
# edges stay near speech level; on real TTS, natural pauses fade to <= 0.012.
DROPOUT_NEAR_ZERO_SAMPLE = 8
DROPOUT_MIN_MS = 10
DROPOUT_EDGE_MS = 5
DROPOUT_CONTEXT_MS = 60
DROPOUT_EDGE_RATIO = 0.25
CLIPPING_ABS_SAMPLE = 32760
# Whitened autocorrelation: on real TTS, clean replies peak at <= 0.0099 and
# echoes at >= 20% of the reply's level at >= 0.0179.
ECHO_MIN_DELAY_S = 0.06
ECHO_MAX_DELAY_S = 0.5
ECHO_WHITENING = 0.75
ECHO_MIN_STRENGTH = 0.015
SPEECH_LEVEL_PERCENTILE = 0.9
NOISE_LEVEL_PERCENTILE = 0.1
NOISE_MIN_SECONDS = 2.0
# A cut-off ends mid-sound: the last 20 ms must stand this far above the
# reply's background noise, not just above a fixed level.
EDGE_ABOVE_NOISE_DB = 10.0
VOLUME_WINDOW_SECONDS = 1.0
VOLUME_PEAK_FRAMES = 5
VOLUME_MIN_WINDOWS = 3

# Lazy import cache
_numpy_mod = None
_numpy_import_error = None


def get_numpy() -> Any:
    """Return the numpy module, or raise ImportError with root cause."""
    global _numpy_mod, _numpy_import_error
    if _numpy_mod is not None:
        return _numpy_mod
    try:
        import numpy

        _numpy_mod = numpy
        return _numpy_mod
    except Exception as e:
        _numpy_import_error = e
        raise ImportError(
            f"numpy is required for audio analysis. Root cause: {e}"
        )


@dataclass
class AudioMeasurements:
    duration: float
    rms_dbfs: float
    speech_dbfs: float
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


def rms(samples: Sequence[int]) -> float:
    np = get_numpy()
    values = np.asarray(samples)
    if values.size == 0:
        return 0.0
    if values.dtype.kind in "iub":
        values = values.astype(np.int64)
        return math.sqrt(int(values @ values) / values.size)
    return math.sqrt(
        sum(sample * sample for sample in values.tolist()) / values.size
    )


def dbfs(level: float) -> float:
    if level <= 0:
        return -96.0
    return 20.0 * math.log10(level / 32768.0)


def frame_samples(
    samples: Sequence[int], sample_rate: int, frame_ms: int = 20
) -> List[Sequence[int]]:
    width = max(1, int(sample_rate * frame_ms / 1000))
    return [
        samples[index : index + width]
        for index in range(0, len(samples), width)
        if len(samples[index : index + width]) >= width // 2
    ]


def _pitch_hz(frame: Sequence[int], sample_rate: int) -> Optional[float]:
    if rms(frame) < audio_utils.DEFAULT_SILENCE_RMS:
        return None
    step = max(1, sample_rate // 8000)
    data = frame[::step]
    rate = sample_rate / step
    min_lag = max(1, int(rate / 400))
    max_lag = min(len(data) // 2, int(rate / 70))
    if max_lag <= min_lag:
        return None
    centered = _centered(data)
    energy = float(centered @ centered)
    if energy <= 0:
        return None
    correlations = _autocorrelation(centered)[min_lag : max_lag + 1]
    best = int(correlations.argmax())
    if float(correlations[best]) / energy < 0.15:
        return None
    return rate / (min_lag + best)


def _centered(data: Sequence[int]) -> Any:
    np = get_numpy()
    values = np.asarray(data)
    if values.dtype.kind in "iub":
        total = int(values.astype(np.int64).sum())
    else:
        total = sum(values.tolist())
    return values - total / len(values)


def _autocorrelation(centered: Any) -> Any:
    """Sum of centered[i] * centered[i + lag], indexed by lag."""
    np = get_numpy()
    return np.correlate(centered, centered, "full")[len(centered) - 1 :]


def pitch_and_periodicity(
    frame: Sequence[int],
    sample_rate: int,
    *,
    min_hz: float = 60.0,
    max_hz: float = 1000.0,
) -> Optional[Tuple[float, float]]:
    """Pitch in Hz and how periodic the frame is (1.0 for a pure tone).

    Searches a wider range than `_pitch_hz`, which stops at 400 Hz: a TTS
    artifact can sit well above a voice. Lags are searched coarsely then
    refined, and the shortest lag close to the best one wins so a
    sub-harmonic is not reported.
    """
    if rms(frame) < audio_utils.DEFAULT_SILENCE_RMS:
        return None
    step = max(1, sample_rate // 8000)
    data = frame[::step]
    rate = sample_rate / step
    min_lag = max(1, int(rate / max_hz))
    max_lag = min(len(data) // 2, int(rate / min_hz))
    if max_lag <= min_lag:
        return None
    np = get_numpy()
    centered = _centered(data)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(centered * centered))
    ).tolist()
    if cumulative[-1] <= 0:
        return None
    products = _autocorrelation(centered).tolist()

    def correlation(lag: int) -> float:
        overlap = len(centered) - lag
        head = cumulative[overlap]
        tail = cumulative[-1] - cumulative[lag]
        if head <= 0 or tail <= 0:
            return 0.0
        return products[lag] / math.sqrt(head * tail)

    coarse = {lag: correlation(lag) for lag in range(min_lag, max_lag + 1, 3)}
    best_coarse = max(coarse.values())
    if best_coarse <= 0:
        return None
    anchor = min(
        lag for lag, value in coarse.items() if value >= 0.9 * best_coarse
    )
    fine = {
        lag: coarse[lag] if lag in coarse else correlation(lag)
        for lag in range(max(min_lag, anchor - 3), min(max_lag, anchor + 3) + 1)
    }
    best_lag = max(fine, key=fine.get)
    best = fine[best_lag]
    left = fine.get(best_lag - 1, best)
    right = fine.get(best_lag + 1, best)
    curvature = left - 2 * best + right
    offset = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
    return rate / (best_lag + offset), best


def _hard_edged_gaps(samples: Sequence[int], sample_rate: int) -> int:
    np = get_numpy()
    values = np.asarray(samples).astype(np.int64)
    quiet = np.abs(values) <= DROPOUT_NEAR_ZERO_SAMPLE
    changes = np.diff(np.concatenate(([0], quiet.astype(np.int8), [0])))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    min_length = sample_rate * DROPOUT_MIN_MS // 1000
    max_length = sample_rate * DROPOUT_MAX_FRAMES * 20 // 1000
    edge = max(1, sample_rate * DROPOUT_EDGE_MS // 1000)
    context = max(1, sample_rate * DROPOUT_CONTEXT_MS // 1000)

    def level(start: int, end: int) -> float:
        window = values[start:end]
        return math.sqrt(int(window @ window) / len(window))

    count = 0
    for start, end in zip(starts.tolist(), ends.tolist()):
        if not (
            min_length <= end - start <= max_length
            and start >= context
            and end + context <= len(values)
        ):
            continue
        before = level(start - context, start)
        after = level(end, end + context)
        if (
            min(before, after) >= audio_utils.DEFAULT_SILENCE_RMS
            and level(start - edge, start) >= DROPOUT_EDGE_RATIO * before
            and level(end, end + edge) >= DROPOUT_EDGE_RATIO * after
        ):
            count += 1
    return count


def echo_delay_and_strength(audio: Audio) -> Tuple[float, float]:
    samples, sample_rate = decode_audio(audio)
    np = get_numpy()
    step = max(1, sample_rate // 8000)
    rate = sample_rate / step
    values = np.asarray(samples[::step], dtype=np.float64)
    values -= values.mean()
    min_lag = int(ECHO_MIN_DELAY_S * rate)
    max_lag = min(len(values), int(ECHO_MAX_DELAY_S * rate))
    if max_lag <= min_lag:
        return 0.0, 0.0
    size = 1 << int(math.ceil(math.log2(2 * len(values))))
    spectrum = np.fft.rfft(values, size)
    magnitude = np.abs(spectrum)
    whitened = magnitude**2 / np.maximum(magnitude, 1e-9) ** (
        2 * ECHO_WHITENING
    )
    correlation = np.fft.irfft(whitened, size)
    if correlation[0] <= 0:
        return 0.0, 0.0
    candidates = correlation[min_lag:max_lag] / correlation[0]
    best = int(candidates.argmax())
    return (min_lag + best) / rate, float(candidates[best])


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))] if ordered else 0.0


def _speech_and_noise_levels(frame_rms: Sequence[float]) -> Tuple[float, float]:
    # Noise is a low percentile of every frame, so noise louder than the
    # silence cutoff still reads as noise instead of as speech.
    levels = [value for value in frame_rms if value > 0]
    voiced = [
        value for value in levels if value >= audio_utils.DEFAULT_SILENCE_RMS
    ]
    return (
        _percentile(voiced or levels, SPEECH_LEVEL_PERCENTILE),
        _percentile(levels, NOISE_LEVEL_PERCENTILE),
    )


def volume_drift_db(audio: Audio) -> Optional[float]:
    samples, sample_rate = decode_audio(audio)
    frame_rms = [rms(frame) for frame in frame_samples(samples, sample_rate)]
    per_window = int(VOLUME_WINDOW_SECONDS * 1000 / 20)
    levels = []
    for start in range(0, len(frame_rms) - per_window + 1, per_window):
        # The loudest few frames give the window's speech level however
        # much of it is pause; a window without that much speech is dropped.
        loudest = sorted(frame_rms[start : start + per_window])[
            -VOLUME_PEAK_FRAMES:
        ]
        if loudest[0] >= audio_utils.DEFAULT_SILENCE_RMS:
            levels.append(dbfs(median(loudest)))
    if len(levels) < VOLUME_MIN_WINDOWS:
        return None
    # Median of pairwise slopes, so one odd window at a word's tail can't
    # tilt the line the way a least-squares fit would.
    slope = median(
        (levels[end] - levels[begin]) / (end - begin)
        for begin in range(len(levels))
        for end in range(begin + 1, len(levels))
    )
    return slope * (len(levels) - 1)


def analyze_audio(audio: Audio) -> AudioMeasurements:
    samples, sample_rate = decode_audio(audio)
    frames = frame_samples(samples, sample_rate)
    frame_rms = [rms(frame) for frame in frames]
    silent = [value < audio_utils.DEFAULT_SILENCE_RMS for value in frame_rms]

    longest_silence = current_silence = 0
    for is_silent in silent:
        if is_silent:
            current_silence += 1
            longest_silence = max(longest_silence, current_silence)
        else:
            current_silence = 0

    pause_levels = [
        value for value, is_silent in zip(frame_rms, silent) if is_silent
    ]
    pause_floor = median(pause_levels) if pause_levels else 0.0
    dropout_events = 0
    index = 1
    while pause_floor >= MIN_PAUSE_FLOOR_RMS and index < len(silent) - 1:
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
            and length <= DROPOUT_MAX_FRAMES
            and min(frame_rms[start:index]) <= DROPOUT_FLOOR_RATIO * pause_floor
        ):
            dropout_events += 1
    if pause_floor < MIN_PAUSE_FLOOR_RMS:
        dropout_events = _hard_edged_gaps(samples, sample_rate)

    speech_level, noise_level = _speech_and_noise_levels(frame_rms)
    snr_db = (
        20.0 * math.log10(max(speech_level, 1.0) / max(noise_level, 1.0))
        if speech_level
        else 0.0
    )

    np = get_numpy()
    values = np.asarray(samples).astype(np.int32)
    negative = values < 0
    crossings = int(np.count_nonzero(negative[:-1] != negative[1:]))
    pitches = [
        pitch
        for frame in frame_samples(samples, sample_rate, frame_ms=40)
        if (pitch := _pitch_hz(frame, sample_rate)) is not None
    ]

    loop_window = max(1, int(sample_rate * 0.25))
    fingerprints = {}
    for start in range(0, len(samples) - loop_window + 1, loop_window):
        window = samples[start : start + loop_window]
        if rms(window) < audio_utils.DEFAULT_SILENCE_RMS:
            continue
        fingerprint = hashlib.blake2b(
            array("h", window).tobytes(), digest_size=8
        ).digest()
        fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
    loop_events = sum(count - 1 for count in fingerprints.values() if count > 2)

    edge_size = max(1, int(sample_rate * 0.02))
    edge_threshold = max(audio_utils.DEFAULT_SILENCE_RMS * 2, 1000)
    end_threshold = edge_threshold
    if len(samples) / sample_rate >= NOISE_MIN_SECONDS:
        # A reply that simply ends inside steady background noise isn't cut.
        end_threshold = max(
            edge_threshold, noise_level * 10 ** (EDGE_ABOVE_NOISE_DB / 20)
        )
    return AudioMeasurements(
        duration=len(samples) / sample_rate,
        rms_dbfs=dbfs(rms(samples)),
        speech_dbfs=dbfs(speech_level),
        clipping_fraction=int(
            np.count_nonzero(np.abs(values) >= CLIPPING_ABS_SAMPLE)
        )
        / len(samples),
        silence_fraction=sum(silent) / len(silent) if silent else 1.0,
        longest_silence_ms=float(longest_silence * 20),
        dropout_events=dropout_events,
        loop_events=loop_events,
        estimated_snr_db=snr_db,
        zero_crossing_rate=crossings / max(1, len(samples) - 1),
        pitch_mean_hz=mean(pitches) if pitches else None,
        pitch_variation_hz=pstdev(pitches) if len(pitches) > 1 else None,
        starts_abruptly=rms(samples[:edge_size]) > edge_threshold,
        ends_abruptly=rms(samples[-edge_size:]) > end_threshold,
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
