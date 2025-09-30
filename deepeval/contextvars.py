from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden


CURRENT_GOLDEN: ContextVar[Optional[Golden]] = ContextVar(
    "CURRENT_GOLDEN", default=None
)


def set_current_golden(golden: Optional[Golden]):
    return CURRENT_GOLDEN.set(golden)


def get_current_golden() -> Optional[Golden]:
    return CURRENT_GOLDEN.get()


def reset_current_golden(token) -> None:
    CURRENT_GOLDEN.reset(token)
