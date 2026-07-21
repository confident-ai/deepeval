from array import array
from typing import Dict, List, Optional, Tuple

TurnRecord = Tuple[int, float, bool]

_FRAME_MS = 20
_CLIP_LEVEL = 32000


def format_measurement(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def compute_percentile(
    sorted_values: List[float], percentile: float
) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (
        rank - low
    )


def summarize_turn_scores(
    records: List[TurnRecord],
    *,
    label: str,
    unit: str,
    limit_description: str,
) -> Tuple[Optional[float], Dict, str]:
    if not records:
        return (
            None,
            {},
            f"No eligible assistant turns to score against the {label} "
            "limit (missing audio/timing on every assistant turn).",
        )

    total = len(records)
    passed = sum(1 for _, _, ok in records if ok)
    values = sorted(value for _, value, _ in records)
    score = passed / total
    breakdown = {
        "eligible": total,
        "passed": passed,
        "mean": sum(values) / len(values),
        "p50": compute_percentile(values, 50),
        "p95": compute_percentile(values, 95),
        "min": values[0],
        "max": values[-1],
    }
    reason = (
        f"{passed}/{total} assistant turn(s) met the {label} limit "
        f"({limit_description}). "
        f"p50={format_measurement(breakdown['p50'])}{unit}, "
        f"p95={format_measurement(breakdown['p95'])}{unit}, "
        f"min={format_measurement(breakdown['min'])}{unit}, "
        f"max={format_measurement(breakdown['max'])}{unit}."
    )
    failed = [(index, value) for index, value, ok in records if not ok]
    if failed:
        worst = sorted(failed, key=lambda r: r[1], reverse=True)[:3]
        offenders = ", ".join(
            f"assistant turn {index} ({format_measurement(value)}{unit})"
            for index, value in worst
        )
        reason += f" Offending turns: {offenders}."
    return score, breakdown, reason


def compute_audio_duration_seconds(audio) -> Optional[float]:
    if audio is None:
        return None
    from deepeval.voice.connectors import audio_utils

    try:
        pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
    except Exception:
        return None
    if not rate:
        return None
    channels = channels or 1
    return (len(pcm) / 2 / channels) / rate


def compute_longest_silence_ms(
    audio, frame_ms: int = _FRAME_MS
) -> Optional[float]:
    if audio is None:
        return None
    from deepeval.voice.connectors import audio_utils

    try:
        pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
    except Exception:
        return None
    if not rate:
        return None
    pcm = audio_utils.downmix_to_mono(pcm, channels or 1)
    longest = current = 0
    for frame in audio_utils.iter_pcm16_frames(pcm, rate, frame_ms):
        if audio_utils.is_silent(frame):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest * frame_ms)


def compute_clipping_percentage(
    audio, clip_level: int = _CLIP_LEVEL
) -> Optional[float]:
    if audio is None:
        return None
    from deepeval.voice.connectors import audio_utils

    try:
        pcm, _, _ = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
    except Exception:
        return None
    samples = array("h")
    samples.frombytes(pcm)
    if len(samples) == 0:
        return None
    clipped = sum(
        1 for sample in samples if sample >= clip_level or sample <= -clip_level
    )
    return 100.0 * clipped / len(samples)
