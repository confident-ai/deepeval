from typing import Any, Optional, Tuple

from deepeval.metrics.voice.utils.analysis import get_numpy, resample

# The agent's voice leaking from the caller's speaker into their microphone
# returns after the round trip; on real phone calls it came back 0.37-0.51 s
# later, while background noise lined up at random delays.
ECHO_RATE = 16000
ECHO_SEARCH_SECONDS = 1.5
ECHO_MIN_DELAY_SECONDS = 0.3
ECHO_MAX_DELAY_SECONDS = 0.6
# How far the delay's correlation peak stands above the typical correlation.
ECHO_MIN_PEAK_RATIO = 12.0
# Copies more than 40 dB below the agent were inaudible to human reviewers.
ECHO_MIN_LEVEL_DB = -40.0
ECHO_MIN_REPLY_SECONDS = 2.0
ECHO_BAND_HZ = (300.0, 3400.0)
ECHO_SEGMENT_SAMPLES = 512


def _delay_and_peak_ratio(agent: Any, caller: Any) -> Tuple[float, float]:
    """The delay at which `agent` best matches `caller`, by phase-only
    cross-correlation, and how sharply that match stands out."""
    np = get_numpy()
    size = 1 << int(np.ceil(np.log2(len(agent) + len(caller))))
    cross = np.fft.rfft(caller, size) * np.conj(np.fft.rfft(agent, size))
    cross /= np.maximum(np.abs(cross), 1e-9)
    correlation = np.fft.irfft(cross, size)[
        : int(ECHO_SEARCH_SECONDS * ECHO_RATE)
    ]
    best = int(np.argmax(correlation))
    typical = float(np.median(np.abs(correlation)))
    ratio = float(correlation[best]) / typical if typical else 0.0
    return best / ECHO_RATE, ratio


def _copy_level_db(agent: Any, caller: Any) -> float:
    """How loud the part of `caller` that matches `agent` is, relative to
    `agent`, in the speech band."""
    np = get_numpy()
    step = ECHO_SEGMENT_SAMPLES // 2
    window = np.hanning(ECHO_SEGMENT_SAMPLES)
    agent_power = caller_power = 0.0
    cross = 0.0
    for start in range(0, len(agent) - ECHO_SEGMENT_SAMPLES + 1, step):
        a = np.fft.rfft(agent[start : start + ECHO_SEGMENT_SAMPLES] * window)
        c = np.fft.rfft(caller[start : start + ECHO_SEGMENT_SAMPLES] * window)
        agent_power = agent_power + np.abs(a) ** 2
        caller_power = caller_power + np.abs(c) ** 2
        cross = cross + a * np.conj(c)
    if isinstance(agent_power, float):
        return float("-inf")
    frequencies = np.fft.rfftfreq(ECHO_SEGMENT_SAMPLES, 1 / ECHO_RATE)
    band = (frequencies >= ECHO_BAND_HZ[0]) & (frequencies <= ECHO_BAND_HZ[1])
    coherence = np.abs(cross[band]) ** 2 / np.maximum(
        agent_power[band] * caller_power[band], 1e-12
    )
    copy_power = float((coherence * caller_power[band]).sum())
    total = float(agent_power[band].sum())
    if copy_power <= 0 or total <= 0:
        return float("-inf")
    return 10 * float(np.log10(copy_power / total))


def speaker_leak(
    agent: Any, caller: Any, sample_rate: int, start: float, end: float
) -> Optional[Tuple[float, float]]:
    """(delay seconds, level dB) when the agent's speech from `start` to `end`
    comes back on the caller's channel, else None."""
    np = get_numpy()
    begin, stop = int(start * sample_rate), int(end * sample_rate)
    reach = min(len(caller), int((end + ECHO_SEARCH_SECONDS) * sample_rate))
    if reach - begin < stop - begin:
        return None
    spoken = resample(agent[begin:stop], sample_rate, ECHO_RATE).astype(
        np.float64
    )
    heard = resample(caller[begin:reach], sample_rate, ECHO_RATE).astype(
        np.float64
    )
    delay, ratio = _delay_and_peak_ratio(
        np.pad(spoken, (0, len(heard) - len(spoken))), heard
    )
    if not (
        ECHO_MIN_DELAY_SECONDS <= delay <= ECHO_MAX_DELAY_SECONDS
        and ratio >= ECHO_MIN_PEAK_RATIO
    ):
        return None
    shift = int(delay * ECHO_RATE)
    level = _copy_level_db(spoken, heard[shift : shift + len(spoken)])
    return (delay, level) if level >= ECHO_MIN_LEVEL_DB else None
