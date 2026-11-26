from typing import Any, Optional, Tuple

from deepeval.metrics.voice.utils.analysis import get_numpy, resample

# The agent's voice leaking from the caller's speaker into their microphone
# returns after the round trip; on real phone calls it came back 0.25-1.7 s
# later, while background noise lined up at random delays.
ECHO_RATE = 16000
ECHO_SEARCH_SECONDS = 2.25
ECHO_MIN_DELAY_SECONDS = 0.15
ECHO_MAX_DELAY_SECONDS = 1.75
# How far the delay's correlation peak stands above the typical correlation.
ECHO_MIN_PEAK_RATIO = 12.0
# Under 2 s of speech the match is noisier: real echoes of short replies
# peaked at 36 or more, chance matches under 20.
ECHO_SHORT_REPLY_SECONDS = 2.0
ECHO_SHORT_REPLY_MIN_PEAK_RATIO = 30.0
# Reviewers of real calls heard copies down to about 16 dB below the agent;
# quieter ones were "barely there" or inaudible.
ECHO_MIN_LEVEL_DB = -17.0
ECHO_MIN_REPLY_SECONDS = 1.0
# A clear echo plays back about as loud as the agent; reviewers called the
# ones more than 8 dB quieter on the caller's line "barely there".
ECHO_MIN_CALLER_LEVEL_DB = -8.0
# When the caller's line is this much louder than the copy, the caller is
# talking and the match is chance or buried under their voice.
ECHO_MAX_OTHER_SOUND_DB = 9.0
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


def _level_db(samples: Any) -> float:
    np = get_numpy()
    power = float(np.mean(samples**2)) if len(samples) else 0.0
    return 10 * float(np.log10(max(power, 1e-9)))


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
    min_ratio = (
        ECHO_SHORT_REPLY_MIN_PEAK_RATIO
        if end - start < ECHO_SHORT_REPLY_SECONDS
        else ECHO_MIN_PEAK_RATIO
    )
    if not (
        ECHO_MIN_DELAY_SECONDS <= delay <= ECHO_MAX_DELAY_SECONDS
        and ratio >= min_ratio
    ):
        return None
    shift = int(delay * ECHO_RATE)
    if len(heard) - shift < len(spoken):
        return None
    copy = heard[shift : shift + len(spoken)]
    level = _copy_level_db(spoken, copy)
    caller_level = _level_db(copy) - _level_db(spoken)
    if (
        level < ECHO_MIN_LEVEL_DB
        or caller_level < ECHO_MIN_CALLER_LEVEL_DB
        or caller_level - level > ECHO_MAX_OTHER_SOUND_DB
    ):
        return None
    return delay, level
