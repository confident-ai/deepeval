import math
from typing import Tuple

from deepeval.metrics.voice.utils.analysis import decode_audio, get_numpy
from deepeval.test_case import Audio

# Whitened autocorrelation: on real TTS, clean replies peak at <= 0.0099 and
# echoes at >= 20% of the reply's level at >= 0.0179.
ECHO_MIN_DELAY_S = 0.06
ECHO_MAX_DELAY_S = 0.5
ECHO_WHITENING = 0.75
ECHO_MIN_STRENGTH = 0.015


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
