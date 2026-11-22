from typing import Optional

from deepeval.metrics.voice.utils.analysis import (
    AudioMeasurements,
    clamp_score,
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
