"""Acoustic measurements used by voice metrics. They need numpy."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
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
# A lost packet is 20 ms; consonants like "p" and "t" leave 10-15 ms gaps.
DROPOUT_MIN_MS = 20
DROPOUT_EDGE_MS = 5
DROPOUT_CONTEXT_MS = 60
DROPOUT_EDGE_RATIO = 0.25
CLIPPING_ABS_SAMPLE = 32760
SPEECH_LEVEL_PERCENTILE = 0.9
NOISE_LEVEL_PERCENTILE = 0.1
NOISE_MIN_SECONDS = 2.0
# A cut-off ends mid-sound: the last 20 ms must stand this far above the
# reply's background noise, not just above a fixed level.
EDGE_ABOVE_NOISE_DB = 10.0
# Where sound stops is found on a 5 ms grid; a 20 ms grid misses cuts that
# land mid-frame.
STOP_FRAME_MS = 5
EDGE_MS = 20
STOP_CONTEXT_SECONDS = 1.0
# On real calls pushed through Opus, mid-sound cuts end at >= 80% of the
# speech before them; natural endings fade below that.
ABRUPT_END_LEVEL_RATIO = 0.8
# A pause sits within this much of the background noise; a wider margin would
# count the quiet ends of words in a noisy reply as pause.
PAUSE_ABOVE_NOISE_DB = 6.0
# Pitch is tracked with YIN (de Cheveigne & Kawahara 2002) on 8 kHz audio.
PITCH_SAMPLE_RATE = 8000
PITCH_FRAME_MS = 40
PITCH_MIN_HZ = 70.0
PITCH_MAX_HZ = 500.0
PITCH_YIN_THRESHOLD = 0.15
# Frames closer than this belong to one stretch of continuous pitch.
PITCH_CONTINUOUS_SEMITONES = 3.0
# The tracker can lock onto the octave for a few frames at a time; a stretch
# that short sitting an octave off its neighbours is folded back.
PITCH_OCTAVE_MAX_FRAMES = 3
PITCH_OCTAVE_TOLERANCE_SEMITONES = 3.0
# 6 semitones in 40 ms is 2.5x the fastest pitch change a speaker can make
# (Xu & Sun 2002).
PITCH_JUMP_SEMITONES = 6.0
PITCH_MIN_VOICED_SECONDS = 1.0
PITCH_MIN_FRAMES = int(PITCH_MIN_VOICED_SECONDS * 1000 / PITCH_FRAME_MS)
# Gaps under 200 ms are part of articulating words, not pauses.
PAUSE_MIN_MS = 200
SPEECH_SPAN_MIN_SECONDS = 2.0
# Clips this close are treated as touching, with no measurable gap.
CONTIGUOUS_CLIP_TOLERANCE_SECONDS = 0.001

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


def resample(samples: Any, sample_rate: int, target_rate: int) -> Any:
    """`samples` as float32 at `target_rate`, by linear interpolation."""
    np = get_numpy()
    values = np.asarray(samples, dtype=np.float32)
    if sample_rate == target_rate or not len(values):
        return values
    count = int(round(len(values) * target_rate / sample_rate))
    return np.interp(
        np.arange(count) * sample_rate / target_rate,
        np.arange(len(values)),
        values,
    ).astype(np.float32)


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
    pitch_mean_hz: Optional[float]
    pitch_variation_hz: Optional[float]
    starts_abruptly: bool
    ends_abruptly: bool
    pitch_median_hz: Optional[float]
    pitch_spread_semitones: Optional[float]
    voiced_pitch_frames: int
    pitch_jumps: int
    speech_span_seconds: float
    longest_pause_ms: float
    pause_fraction: float
    # Where in the clip each problem sits, in seconds from its start:
    # (offset, duration) spans, and the time of each pitch jump.
    dropout_spans: List[Tuple[float, float]] = field(default_factory=list)
    loop_spans: List[Tuple[float, float]] = field(default_factory=list)
    pitch_jump_times: List[float] = field(default_factory=list)
    longest_pause_span: Optional[Tuple[float, float]] = None
    sound_end_seconds: Optional[float] = None


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


def semitones(hz: float, reference_hz: float) -> float:
    return 12.0 * math.log2(hz / reference_hz)


def _downsampled(samples: Sequence[int], sample_rate: int) -> Tuple[Any, float]:
    np = get_numpy()
    values = np.asarray(samples, dtype=np.float64)
    step = sample_rate // PITCH_SAMPLE_RATE
    if step <= 1:
        return values, float(sample_rate)
    # Averaging each block removes what the lower rate can't hold.
    usable = len(values) // step * step
    return values[:usable].reshape(-1, step).mean(axis=1), sample_rate / step


def _yin_pitch_hz(frame: Any, rate: float) -> Optional[float]:
    np = get_numpy()
    max_lag = int(rate / PITCH_MIN_HZ)
    min_lag = max(2, int(rate / PITCH_MAX_HZ))
    window = len(frame) - max_lag
    if window <= 0:
        return None
    centered = frame - frame.mean()
    level = math.sqrt(float(centered @ centered) / len(centered))
    if level < audio_utils.DEFAULT_SILENCE_RMS:
        return None
    energy = np.concatenate(([0.0], np.cumsum(centered * centered)))
    lags = np.arange(max_lag + 1)
    products = np.correlate(
        centered[: window + max_lag], centered[:window], "valid"
    )[: max_lag + 1]
    difference = (
        energy[window] + energy[lags + window] - energy[lags] - 2 * products
    )
    normalized = np.ones(max_lag + 1)
    normalized[1:] = (
        difference[1:] * lags[1:] / np.maximum(np.cumsum(difference[1:]), 1e-9)
    )
    below = np.nonzero(normalized[min_lag:max_lag] < PITCH_YIN_THRESHOLD)[0]
    if below.size == 0:
        return None
    lag = min_lag + int(below[0])
    while lag + 1 < max_lag and normalized[lag + 1] < normalized[lag]:
        lag += 1
    left, best, right = normalized[lag - 1 : lag + 2]
    curvature = left - 2 * best + right
    offset = 0.5 * (left - right) / curvature if curvature > 0 else 0.0
    return float(rate / (lag + offset))


def _pitch_stretches(track: Sequence[Optional[float]]) -> List[Tuple[int, int]]:
    stretches = []
    index = 0
    while index < len(track):
        if track[index] is None:
            index += 1
            continue
        end = index + 1
        while (
            end < len(track)
            and track[end] is not None
            and abs(semitones(track[end], track[end - 1]))
            < PITCH_CONTINUOUS_SEMITONES
        ):
            end += 1
        stretches.append((index, end))
        index = end
    return stretches


def _without_octave_errors(
    track: Sequence[Optional[float]],
) -> List[Optional[float]]:
    track = list(track)
    # A second pass catches an error that sat next to another one.
    for _ in range(2):
        stretches = _pitch_stretches(track)
        for position, (start, end) in enumerate(stretches):
            if end - start > PITCH_OCTAVE_MAX_FRAMES:
                continue
            level = median(track[start:end])
            neighbours = []
            if position > 0 and stretches[position - 1][1] == start:
                neighbours.append(track[start - 1])
            if (
                position + 1 < len(stretches)
                and stretches[position + 1][0] == end
            ):
                neighbours.append(track[end])
            offsets = [semitones(level, pitch) for pitch in neighbours]
            if (
                offsets
                and all(
                    abs(abs(offset) - 12.0) <= PITCH_OCTAVE_TOLERANCE_SEMITONES
                    for offset in offsets
                )
                and len({offset > 0 for offset in offsets}) == 1
            ):
                factor = 0.5 if offsets[0] > 0 else 2.0
                for index in range(start, end):
                    track[index] *= factor
    return track


def pitch_track(
    samples: Sequence[int], sample_rate: int
) -> List[Optional[float]]:
    """Pitch of each 40 ms frame, None where it is unvoiced."""
    data, rate = _downsampled(samples, sample_rate)
    width = int(rate * PITCH_FRAME_MS / 1000)
    return _without_octave_errors(
        [
            _yin_pitch_hz(data[start : start + width], rate)
            for start in range(0, len(data) - width + 1, width)
        ]
    )


def count_pitch_jumps(track: Sequence[Optional[float]]) -> int:
    return len(pitch_jump_frames(track))


def pitch_jump_frames(track: Sequence[Optional[float]]) -> List[int]:
    """Frames where pitch steps 6+ semitones between two stretches of steady
    pitch; a lone frame on either side is more likely a tracking slip than a
    crack."""
    stretches = _pitch_stretches(track)
    return [
        next_start
        for (start, end), (next_start, next_end) in zip(
            stretches, stretches[1:]
        )
        if end == next_start
        and end - start >= 2
        and next_end - next_start >= 2
        and abs(semitones(track[next_start], track[end - 1]))
        >= PITCH_JUMP_SEMITONES
    ]


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

    Searches a wider range than `pitch_track`, which stops at 500 Hz: a TTS
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


def _hard_edged_gaps(
    samples: Sequence[int], sample_rate: int
) -> List[Tuple[float, float]]:
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

    gaps = []
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
            gaps.append((start / sample_rate, (end - start) / sample_rate))
    return gaps


LOOP_MIN_SECONDS = 0.2
LOOP_MAX_SECONDS = 2.0
LOOP_RATE = 8000
LOOP_FRAME_MS = 40
LOOP_MIN_LAG_CORRELATION = 0.2
LOOP_MIN_FRAME_CORRELATION = 0.9
LOOP_CANDIDATE_LAGS = 5
LOOP_MIN_LEVEL_SPREAD_DB = 3.0


def _loops(
    samples: Sequence[int], sample_rate: int
) -> List[Tuple[float, float, int]]:
    np = get_numpy()
    values, rate = _downsampled(samples, sample_rate)
    if rate > LOOP_RATE * 1.5:
        step = int(rate // LOOP_RATE)
        values, rate = values[::step], rate / step
    values = values - values.mean() if len(values) else values
    min_lag = int(LOOP_MIN_SECONDS * rate)
    max_lag = min(int(LOOP_MAX_SECONDS * rate), len(values) // 3)
    if max_lag <= min_lag:
        return []
    size = 1 << int(math.ceil(math.log2(2 * len(values))))
    spectrum = np.fft.rfft(values, size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), size)
    if autocorrelation[0] <= 0:
        return []
    overlap = len(values) - np.arange(len(autocorrelation[: max_lag + 1]))
    normalised = (
        autocorrelation[: max_lag + 1]
        / autocorrelation[0]
        * len(values)
        / np.maximum(overlap, 1)
    )
    window = normalised[min_lag : max_lag + 1]
    candidates = [
        min_lag + int(index)
        for index in np.argsort(window)[::-1][: LOOP_CANDIDATE_LAGS * 4]
        if window[index] >= LOOP_MIN_LAG_CORRELATION
    ]
    frame = int(rate * LOOP_FRAME_MS / 1000)
    loops: List[Tuple[float, float, int]] = []
    tried: List[int] = []
    for lag in candidates:
        if any(abs(lag - other) < frame for other in tried):
            continue
        tried.append(lag)
        if len(tried) > LOOP_CANDIDATE_LAGS:
            break
        frames = (len(values) - lag) // frame
        matches: List[Optional[bool]] = []
        for index in range(frames):
            a = values[index * frame : (index + 1) * frame]
            b = values[index * frame + lag : (index + 1) * frame + lag]
            quiet = (
                max(
                    math.sqrt(float(a @ a) / frame),
                    math.sqrt(float(b @ b) / frame),
                )
                < audio_utils.DEFAULT_SILENCE_RMS
            )
            energy = math.sqrt(float(a @ a) * float(b @ b))
            matches.append(
                None
                if quiet
                else energy > 0
                and float(a @ b) / energy >= LOOP_MIN_FRAME_CORRELATION
            )
        index = 0
        while index < frames:
            if matches[index] is not True:
                index += 1
                continue
            run_start = last_match = index
            while index < frames and matches[index] is not False:
                if matches[index]:
                    last_match = index
                index += 1
            index = last_match + 1
            run = (index - run_start) * frame
            if run + frame < 2 * lag:
                continue
            levels = [
                dbfs(math.sqrt(float(chunk @ chunk) / frame))
                for chunk in (
                    values[position : position + frame]
                    for position in range(
                        run_start * frame, run_start * frame + lag, frame
                    )
                )
                if len(chunk) == frame
            ]
            if (
                not levels
                or max(levels) - min(levels) < LOOP_MIN_LEVEL_SPREAD_DB
            ):
                continue
            offset = (run_start * frame + lag) / rate
            duration = run / rate
            if any(
                offset < existing + length and existing < offset + duration
                for existing, length, _ in loops
            ):
                continue
            loops.append((offset, duration, max(1, round(run / lag))))
    return sorted(loops)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))] if ordered else 0.0


def speech_and_noise_levels(frame_rms: Sequence[float]) -> Tuple[float, float]:
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


def _snr_from_quiet_frames(frame_rms: Sequence[float]) -> float:
    # Too short for a pause-based noise level; keeps the earlier estimate so
    # short clips score as they always have.
    voiced = [
        value
        for value in frame_rms
        if value > 0 and value >= audio_utils.DEFAULT_SILENCE_RMS
    ]
    quiet = [
        value
        for value in frame_rms
        if value > 0 and value < audio_utils.DEFAULT_SILENCE_RMS
    ]
    speech_level = _percentile(voiced, 0.75)
    noise_level = (
        _percentile(quiet, 0.25) if quiet else max(1.0, speech_level / 100.0)
    )
    if not speech_level:
        return 0.0
    return 20.0 * math.log10(max(speech_level, 1.0) / max(noise_level, 1.0))


def _edge_threshold() -> float:
    return max(audio_utils.DEFAULT_SILENCE_RMS * 2, 1000)


def ends_abruptly(
    samples: Sequence[int], sample_rate: int, noise_level: float
) -> bool:
    """Whether the sound stops at near full volume instead of fading out."""
    return sound_stop(samples, sample_rate, noise_level)[0]


def sound_stop(
    samples: Sequence[int], sample_rate: int, noise_level: float
) -> Tuple[bool, Optional[float]]:
    """Whether the sound stops at near full volume, and where it stops in
    seconds from the start of `samples`."""
    np = get_numpy()
    width = max(1, sample_rate * STOP_FRAME_MS // 1000)
    values = np.asarray(samples, dtype=np.float64)
    usable = len(values) // width * width
    if not usable:
        return False, None
    levels = np.sqrt((values[:usable].reshape(-1, width) ** 2).mean(axis=1))
    sounding = np.flatnonzero(levels >= audio_utils.DEFAULT_SILENCE_RMS)
    if not sounding.size:
        return False, None
    stop = int(sounding[-1]) + 1
    stop_seconds = stop * width / sample_rate
    edge_frames = EDGE_MS // STOP_FRAME_MS
    edge = math.sqrt(
        float((levels[max(0, stop - edge_frames) : stop] ** 2).mean())
    )
    context = levels[
        max(0, stop - int(STOP_CONTEXT_SECONDS * 1000 / STOP_FRAME_MS)) : stop
    ]
    speech_level = float(
        np.median(context[context >= audio_utils.DEFAULT_SILENCE_RMS])
    )
    floor = max(
        _edge_threshold(), noise_level * 10 ** (EDGE_ABOVE_NOISE_DB / 20)
    )
    return (
        edge > floor and edge >= ABRUPT_END_LEVEL_RATIO * speech_level,
        stop_seconds,
    )


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
    dropout_spans = []
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
            dropout_spans.append((start * 0.02, length * 0.02))
    if pause_floor < MIN_PAUSE_FLOOR_RMS:
        dropout_spans = _hard_edged_gaps(samples, sample_rate)

    speech_level, noise_level = speech_and_noise_levels(frame_rms)
    if len(samples) / sample_rate >= NOISE_MIN_SECONDS:
        snr_db = (
            20.0 * math.log10(max(speech_level, 1.0) / max(noise_level, 1.0))
            if speech_level
            else 0.0
        )
    else:
        snr_db = _snr_from_quiet_frames(frame_rms)

    np = get_numpy()
    values = np.asarray(samples).astype(np.int32)
    track = pitch_track(samples, sample_rate)
    jump_frames = pitch_jump_frames(track)
    pitches = [pitch for pitch in track if pitch is not None]
    pitch_median = median(pitches) if pitches else None
    pitch_spread = (
        # Scaled median deviation: a standard deviation that one cracked
        # word can't move.
        1.4826
        * median(abs(semitones(pitch, pitch_median)) for pitch in pitches)
        if len(pitches) > 1
        else None
    )

    loops = _loops(samples, sample_rate)
    loop_events = sum(extra for _, _, extra in loops)
    loop_spans = [(offset, duration) for offset, duration, _ in loops]

    edge_size = max(1, int(sample_rate * 0.02))
    edge_threshold = _edge_threshold()
    end_noise_level = 0.0
    speaking = audio_utils.DEFAULT_SILENCE_RMS
    if len(samples) / sample_rate >= NOISE_MIN_SECONDS:
        # A reply that simply ends inside steady background noise isn't cut.
        end_noise_level = noise_level
        pause_ceiling = noise_level * 10 ** (PAUSE_ABOVE_NOISE_DB / 20)
        # With no quieter stretch to call noise, the "noise" is the speech.
        if pause_ceiling < speech_level:
            speaking = max(speaking, pause_ceiling)
    spoken = [
        index for index, value in enumerate(frame_rms) if value >= speaking
    ]
    pauses = []
    pause_spans = []
    if spoken:
        current = 0
        for index in range(spoken[0], spoken[-1] + 1):
            if frame_rms[index] < speaking:
                current += 1
                continue
            if current * 20 >= PAUSE_MIN_MS:
                pauses.append(current * 20)
                pause_spans.append(((index - current) * 0.02, current * 0.02))
            current = 0
    speech_span_ms = (spoken[-1] - spoken[0] + 1) * 20 if spoken else 0
    stop = sound_stop(samples, sample_rate, end_noise_level)
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
        dropout_events=len(dropout_spans),
        loop_events=loop_events,
        estimated_snr_db=snr_db,
        pitch_mean_hz=mean(pitches) if pitches else None,
        pitch_variation_hz=pstdev(pitches) if len(pitches) > 1 else None,
        starts_abruptly=rms(samples[:edge_size]) > edge_threshold,
        ends_abruptly=stop[0],
        pitch_median_hz=pitch_median,
        pitch_spread_semitones=pitch_spread,
        voiced_pitch_frames=len(pitches),
        pitch_jumps=len(jump_frames),
        speech_span_seconds=speech_span_ms / 1000,
        longest_pause_ms=float(max(pauses, default=0)),
        pause_fraction=sum(pauses) / speech_span_ms if speech_span_ms else 0.0,
        dropout_spans=dropout_spans,
        loop_spans=loop_spans,
        pitch_jump_times=[
            frame * PITCH_FRAME_MS / 1000 for frame in jump_frames
        ],
        longest_pause_span=(
            max(pause_spans, key=lambda span: span[1]) if pause_spans else None
        ),
        sound_end_seconds=stop[1],
    )
