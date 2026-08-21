"""Interruption level policy for duplex voice simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # `Persona` declares these, so they live with it in `deepeval.dataset`.
    # Importing that package here would be a cycle — `deepeval.models` pulls in
    # `deepeval.voice`, and `deepeval.dataset` pulls in `deepeval.models` — so
    # they are re-exported through `__getattr__` below instead. Annotations in
    # this module are strings (`from __future__ import annotations`), so nothing
    # else needs them at import time.
    from deepeval.dataset.golden import (
        InterruptionBehavior,
        InterruptionLevel,
        OverlapBehavior,
    )

_PERSONA_TYPES = (
    "InterruptionBehavior",
    "InterruptionLevel",
    "OverlapBehavior",
)

__all__ = [
    "InterruptionBehavior",
    "InterruptionLevel",
    "OverlapBehavior",
    "InterruptionPolicy",
    "interruption_policy",
    "should_poll_judge",
]


@dataclass(frozen=True)
class InterruptionPolicy:
    """Throttle / cap knobs derived from `interruption_level`.

    Prompt prose for each level lives in
    ``deepeval/simulator/templates/interruption_bias_*.txt`` and is rendered
    via the simulator template bundle — not stored here.
    """

    level: InterruptionLevel
    # Minimum growth in partial transcript chars before re-judging.
    min_partial_delta_chars: int
    # Minimum wall time between judge calls (seconds).
    min_poll_interval_s: float
    # Soft cap on barge attempts per conversation.
    max_barges_per_conversation: int
    # Soft cap on barge attempts against a single agent utterance.
    max_barges_per_agent_turn: int


def interruption_policy(
    level: Optional[InterruptionLevel],
) -> Optional[InterruptionPolicy]:
    if level is None:
        return None
    if level == "rare":
        return InterruptionPolicy(
            level=level,
            min_partial_delta_chars=80,
            min_poll_interval_s=2.0,
            max_barges_per_conversation=1,
            max_barges_per_agent_turn=1,
        )
    if level == "normal":
        return InterruptionPolicy(
            level=level,
            min_partial_delta_chars=40,
            min_poll_interval_s=1.0,
            max_barges_per_conversation=4,
            max_barges_per_agent_turn=2,
        )
    if level == "frequent":
        return InterruptionPolicy(
            level=level,
            min_partial_delta_chars=20,
            min_poll_interval_s=0.5,
            max_barges_per_conversation=8,
            max_barges_per_agent_turn=3,
        )
    raise ValueError(
        f"Invalid interruption_level {level!r}; expected None, "
        "'rare', 'normal', or 'frequent'."
    )


def should_poll_judge(
    *,
    policy: InterruptionPolicy,
    partial_transcript: str,
    last_judged_len: int,
    last_judge_at: Optional[float],
    now: float,
    barges_this_conversation: int,
    barges_this_agent_turn: int,
) -> bool:
    if barges_this_conversation >= policy.max_barges_per_conversation:
        return False
    if barges_this_agent_turn >= policy.max_barges_per_agent_turn:
        return False
    delta = len(partial_transcript) - last_judged_len
    if delta < policy.min_partial_delta_chars:
        return False
    if (
        last_judge_at is not None
        and (now - last_judge_at) < policy.min_poll_interval_s
    ):
        return False
    return True


def __getattr__(name: str):
    if name in _PERSONA_TYPES:
        from deepeval.dataset import golden

        return getattr(golden, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
