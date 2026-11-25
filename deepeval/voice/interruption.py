"""Which agent turns the simulated caller interrupts."""

from __future__ import annotations

import random
from typing import Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # `Persona` declares these, so they live with it in `deepeval.dataset`.
    # Importing that package here would be a cycle — `deepeval.models` pulls in
    # `deepeval.voice`, and `deepeval.dataset` pulls in `deepeval.models` — so
    # they are re-exported through `__getattr__` below instead. Annotations in
    # this module are strings (`from __future__ import annotations`), so nothing
    # else needs them at import time.
    from deepeval.dataset.golden import (
        InterruptionBehavior,
        OverlapBehavior,
    )

_PERSONA_TYPES = (
    "InterruptionBehavior",
    "OverlapBehavior",
)

__all__ = [
    "InterruptionBehavior",
    "OverlapBehavior",
    "InterruptionSchedule",
]


class InterruptionSchedule:
    """Counts agent turns and says which ones the caller cuts in on.

    Every `every` agent turns (or a number drawn from the `(fewest, most)`
    range) the caller interrupts. A turn that could not be interrupted, such
    as one too short to cut into, leaves the interruption owed to the next.
    """

    def __init__(
        self,
        every: Union[int, Tuple[int, int]] = (2, 3),
        *,
        seed: Optional[int] = None,
    ):
        from deepeval.dataset.golden import _turn_range

        self.every = _turn_range(every)
        self._rng = random.Random(seed)
        self._turns_left = self._draw()

    def _draw(self) -> int:
        return self._rng.randint(*self.every)

    @property
    def interrupts_next_reply(self) -> bool:
        return self._turns_left <= 1

    def on_agent_reply_start(self) -> bool:
        self._turns_left = max(0, self._turns_left - 1)
        return self._turns_left == 0

    def on_interruption_landed(self) -> None:
        self._turns_left = self._draw()


def __getattr__(name: str):
    if name in _PERSONA_TYPES:
        from deepeval.dataset import golden

        return getattr(golden, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
