from typing import Optional

from deepeval.metrics.voice.utils.analysis import (
    SPEECH_SPAN_MIN_SECONDS,
    AudioMeasurements,
)


def _spoken_word_count(content: str) -> int:
    # Digits are read out one by one, so "555-1234" is seven spoken words.
    return sum(
        sum(character.isdigit() for character in token)
        + any(character.isalpha() for character in token)
        for token in content.split()
    )


def speech_span_rate_wpm(
    content: str, measurements: AudioMeasurements
) -> Optional[float]:
    """Spoken words per minute from first to last speech, pauses included,
    the way conversational rates are usually measured."""
    words = _spoken_word_count(content)
    if words == 0 or measurements.speech_span_seconds < SPEECH_SPAN_MIN_SECONDS:
        return None
    return words / (measurements.speech_span_seconds / 60.0)
