from contextvars import ContextVar
from typing import Optional


CURRENT_GOLDEN: ContextVar[Optional[object]] = ContextVar(
    "CURRENT_GOLDEN", default=None
)


def set_current_golden(golden: Optional[object]):
    return CURRENT_GOLDEN.set(golden)


def get_current_golden() -> Optional[object]:
    return CURRENT_GOLDEN.get()


def reset_current_golden(token) -> None:
    CURRENT_GOLDEN.reset(token)
