"""Turn-detection presets for voice connectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TurnDetection = Literal["eager", "balanced", "patient"]

_TURN_DETECTION_LEVELS = ("eager", "balanced", "patient")

__all__ = [
    "TurnDetection",
    "TurnDetectionTiming",
    "turn_detection_timing",
]


@dataclass(frozen=True)
class TurnDetectionTiming:
    """Timing selected by a connector's ``turn_detection`` level.

    Nothing in an audio stream announces that the agent has finished speaking,
    so the end of its turn is inferred from silence. These are the two numbers
    that inference needs, chosen together rather than tuned individually.
    """

    level: TurnDetection
    # Quiet since the last speech before the agent is taken to have finished.
    end_of_turn_silence_ms: int
    # Hard ceiling on waiting, so an agent that never goes quiet cannot hang
    # the simulation. In duplex this bounds the whole exchange, barges included.
    max_turn_timeout_s: float


def turn_detection_timing(level: TurnDetection) -> TurnDetectionTiming:
    if level == "eager":
        # For agents that answer in one breath. Reclaims the floor quickly, at
        # the risk of clipping an agent that stops to think.
        return TurnDetectionTiming(
            level=level,
            end_of_turn_silence_ms=500,
            max_turn_timeout_s=20.0,
        )
    if level == "balanced":
        return TurnDetectionTiming(
            level=level,
            end_of_turn_silence_ms=800,
            max_turn_timeout_s=30.0,
        )
    if level == "patient":
        # For agents that pause mid-reply — to think, to look something up, or
        # just to breathe. Waits through those pauses rather than cutting the
        # reply short and discarding the rest of it.
        return TurnDetectionTiming(
            level=level,
            end_of_turn_silence_ms=2500,
            max_turn_timeout_s=120.0,
        )
    raise ValueError(
        f"Invalid turn_detection {level!r}; expected 'eager', 'balanced', "
        "or 'patient'."
    )
