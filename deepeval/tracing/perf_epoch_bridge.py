# =====================================
#   perf_epoch_bridge.py
# =====================================
"""
Bi-directional conversion utilities between UNIX-epoch nanoseconds
and the Python perf_counter() clock.

Usage:
    >>> import perf_epoch_bridge as peb
    >>> peb.init_clock_bridge()
    >>> span_start = peb.epoch_nanos_to_perf_seconds(start_time_unix_nano)
    >>> span_end   = peb.epoch_nanos_to_perf_seconds(end_time_unix_nano)
    >>> duration   = span_end - span_start
"""

from __future__ import annotations
import time
from typing import Final, Union

# Module globals are initialised exactly once.
_anchor_perf_ns: Union[int, None] = None
_anchor_wall_ns: Union[int, None] = None
_offset_ns: Union[int, None] = None


def init_clock_bridge() -> None:
    """Capture simultaneous perf & wall-clock samples and compute offset."""
    global _anchor_perf_ns, _anchor_wall_ns, _offset_ns

    # Capture as closely together as possible
    _anchor_perf_ns = time.perf_counter_ns()
    _anchor_wall_ns = time.time_ns()
    _offset_ns = _anchor_perf_ns - _anchor_wall_ns


def epoch_nanos_to_perf_seconds(epoch_ns: int) -> float:
    """Translate a UNIX epoch (ns) timestamp onto perf_counter() seconds."""
    if _offset_ns is None:
        raise RuntimeError("init_clock_bridge() must be called first!")
    return (epoch_ns + _offset_ns) / 1_000_000_000.0


def perf_seconds_now() -> float:
    """Return current perf_counter() reading (seconds)."""
    return time.perf_counter()


# Optional: reverse conversion (perf → epoch)
def perf_seconds_to_epoch_nanos(perf_sec: float) -> int:
    """Translate a perf_counter() float back to epoch nanoseconds."""
    if _offset_ns is None:
        raise RuntimeError("init_clock_bridge() must be called first!")
    return int(perf_sec * 1_000_000_000) - _offset_ns
