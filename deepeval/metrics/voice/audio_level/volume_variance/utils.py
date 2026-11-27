from statistics import correlation, median
from typing import List, Optional, Sequence

from deepeval.metrics.voice.utils.analysis import (
    dbfs,
    decode_audio,
    frame_samples,
    rms,
)
from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils

VOLUME_WINDOW_SECONDS = 1.0
VOLUME_PEAK_FRAMES = 5
VOLUME_MIN_SPEECH_FRAMES = 15
VOLUME_MIN_WINDOWS = 6
# How consistently the level moves one way (rank correlation with time).
# Speech rises and falls from one second to the next; a fade keeps going.
VOLUME_STEADY_TREND = 0.8


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def volume_drift_db(audio: Audio) -> Optional[float]:
    """How far the reply's volume fades (negative) or swells from start to
    end: 0.0 when it doesn't keep moving one way, None when the reply is
    too short to tell."""
    samples, sample_rate = decode_audio(audio)
    frame_rms = [rms(frame) for frame in frame_samples(samples, sample_rate)]
    per_window = int(VOLUME_WINDOW_SECONDS * 1000 / 20)
    times = []
    levels = []
    for start in range(0, len(frame_rms) - per_window + 1, per_window):
        window = frame_rms[start : start + per_window]
        # A window holding only a word's tail would read as a fake fade.
        speech_frames = sum(
            value >= audio_utils.DEFAULT_SILENCE_RMS for value in window
        )
        if speech_frames >= VOLUME_MIN_SPEECH_FRAMES:
            # The loudest few frames give the speech level however much of
            # the window is pause.
            times.append(float(start // per_window))
            levels.append(dbfs(median(sorted(window)[-VOLUME_PEAK_FRAMES:])))
    if len(levels) < VOLUME_MIN_WINDOWS:
        return None
    if abs(correlation(times, _ranks(levels))) < VOLUME_STEADY_TREND:
        return 0.0
    # Median of pairwise slopes, so one odd window can't tilt the line the
    # way a least-squares fit would.
    slope = median(
        (levels[end] - levels[begin]) / (times[end] - times[begin])
        for begin in range(len(levels))
        for end in range(begin + 1, len(levels))
    )
    return slope * (times[-1] - times[0])
