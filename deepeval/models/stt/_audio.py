"""Audio inspection shared by the STT models."""

from __future__ import annotations

import io
import wave
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from deepeval.test_case import Audio


def audio_duration_seconds(audio: "Audio") -> Optional[float]:
    """How long `audio` runs, or `None` when it cannot be determined.

    Needed because several providers price transcription by the minute but
    report usage only sometimes, leaving the duration to be measured locally.
    Anything that is not a readable WAV yields `None` rather than a guess, so a
    cost is never invented from a number we do not have.
    """
    if audio.duration is not None:
        return audio.duration
    try:
        with wave.open(io.BytesIO(audio.get_bytes()), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate:
                return frames / float(rate)
    except (wave.Error, EOFError, ValueError):
        pass
    return None
