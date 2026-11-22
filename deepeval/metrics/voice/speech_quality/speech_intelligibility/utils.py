from deepeval.metrics.voice.utils.analysis import (
    AudioMeasurements,
    clamp_score,
)


def intelligibility_score(measurements: AudioMeasurements) -> float:
    snr = clamp_score((measurements.estimated_snr_db - 3.0) / 22.0)
    volume = clamp_score(1.0 - abs(measurements.rms_dbfs + 20.0) / 35.0)
    clipping = clamp_score(1.0 - measurements.clipping_fraction * 30.0)
    dropouts = clamp_score(1.0 - measurements.dropout_events * 0.12)
    return clamp_score(
        0.4 * snr + 0.25 * volume + 0.2 * clipping + 0.15 * dropouts
    )
