"""
Regression tests for issue #2952 – event-loop safety in deepeval.

The safe_asyncio_run function is tested in isolation (without importing the
full deepeval package) so these tests have no heavy dependency requirements.

Covers:
  1. Plain sync Python process (no loop running)
  2. Async context via pytest-asyncio (loop already running)
  3. Simulated "Jupyter / FastAPI" scenario (nest_asyncio + running loop)
  4. Multiple sequential calls don't deadlock
"""

import asyncio
import importlib.util
import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import safe_asyncio_run directly from the source file, bypassing the full
# deepeval __init__.py chain (which requires sentry_sdk and other heavy deps).
# ---------------------------------------------------------------------------
_utils_path = (
    Path(__file__).parents[1] / "deepeval" / "models" / "llms" / "utils.py"
)
_spec = importlib.util.spec_from_file_location(
    "deepeval_llm_utils", _utils_path
)
_utils_mod = importlib.util.module_from_spec(_spec)

# Stub out deepeval.errors so the lone `from deepeval.errors import ...` line
# inside utils.py doesn't trigger the full package import chain.
_fake_errors = _types.ModuleType("deepeval.errors")


class _DeepEvalError(Exception):
    pass


_fake_errors.DeepEvalError = _DeepEvalError
sys.modules.setdefault("deepeval", _types.ModuleType("deepeval"))
sys.modules["deepeval.errors"] = _fake_errors
_spec.loader.exec_module(_utils_mod)
safe_asyncio_run = _utils_mod.safe_asyncio_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _add(a: int, b: int) -> int:
    """Trivial async coroutine used as a test fixture."""
    await asyncio.sleep(0)  # yield once to exercise real async scheduling
    return a + b


# ---------------------------------------------------------------------------
# 1. Sync execution in a plain Python process (no loop running)
# ---------------------------------------------------------------------------


def test_safe_asyncio_run_no_loop():
    """safe_asyncio_run works when called from a plain sync context."""
    result = safe_asyncio_run(_add(1, 2))
    assert result == 3


# ---------------------------------------------------------------------------
# 2. Async execution under pytest-asyncio (loop is already running)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safe_asyncio_run_inside_running_loop():
    """safe_asyncio_run must NOT raise RuntimeError when a loop is active."""
    # This exercises the nest_asyncio branch that was broken in the original
    # implementation (it tried asyncio.run() first, which raises here).
    result = safe_asyncio_run(_add(3, 4))
    assert result == 7


# ---------------------------------------------------------------------------
# 3. Notebook / FastAPI-like scenario: nest_asyncio pre-applied, loop active
# ---------------------------------------------------------------------------


def test_safe_asyncio_run_with_nest_asyncio_pre_applied():
    """
    Simulate the Jupyter / FastAPI scenario where nest_asyncio is already
    applied and a loop is running.

    We spin up a fresh event loop, apply nest_asyncio to it, and call
    safe_asyncio_run from *within* that running loop – exactly the pattern
    that triggered the original RuntimeError.
    """
    import nest_asyncio

    loop = asyncio.new_event_loop()
    nest_asyncio.apply(loop)

    async def _inner():
        return safe_asyncio_run(_add(10, 20))

    result = loop.run_until_complete(_inner())
    loop.close()
    assert result == 30


# ---------------------------------------------------------------------------
# 4. Multiple sequential calls don't deadlock
# ---------------------------------------------------------------------------


def test_safe_asyncio_run_multiple_calls_no_deadlock():
    """Calling safe_asyncio_run repeatedly in the same process doesn't hang."""
    for i in range(5):
        assert safe_asyncio_run(_add(i, i)) == i * 2


@pytest.mark.asyncio
async def test_safe_asyncio_run_multiple_calls_inside_running_loop():
    """Same check but from within a running async context."""
    for i in range(5):
        assert safe_asyncio_run(_add(i, i)) == i * 2
