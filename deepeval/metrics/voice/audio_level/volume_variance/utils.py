from statistics import median
from typing import Optional

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
VOLUME_MIN_WINDOWS = 3


def volume_drift_db(audio: Audio) -> Optional[float]:
    samples, sample_rate = decode_audio(audio)
    frame_rms = [rms(frame) for frame in frame_samples(samples, sample_rate)]
    per_window = int(VOLUME_WINDOW_SECONDS * 1000 / 20)
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
            levels.append(dbfs(median(sorted(window)[-VOLUME_PEAK_FRAMES:])))
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
