"""
Regression test: the crewai integration must import cleanly even when crewai is
not installed, and every public entry point must raise a *helpful* ImportError
when actually used.

crewai's absence is simulated via an import blocker so this runs deterministically
regardless of whether crewai happens to be installed in the test environment.
"""

import sys
import importlib
from contextlib import contextmanager

import pytest


class _BlockModules:
    """A meta_path finder that makes the given top-level packages look absent."""

    def __init__(self, *blocked):
        self._blocked = blocked

    def find_spec(self, name, path=None, target=None):
        if name in self._blocked or name.startswith(
            tuple(p + "." for p in self._blocked)
        ):
            raise ModuleNotFoundError(f"blocked for test: {name}")
        return None


@contextmanager
def dependency_absent(dep_prefix, integration_module):
    """Run a block with ``dep_prefix`` forced unavailable, clearing and then
    restoring cached ``dep_prefix``/``integration_module`` modules so the
    integration re-imports fresh and later tests are unaffected."""
    watch = (dep_prefix, integration_module)
    saved = {k: v for k, v in sys.modules.items() if k.startswith(watch)}
    for key in list(sys.modules):
        if key.startswith(watch):
            del sys.modules[key]

    finder = _BlockModules(dep_prefix)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for key in list(sys.modules):
            if key.startswith(watch):
                del sys.modules[key]
        sys.modules.update(saved)


def test_crewai_integration_imports_without_crewai():
    """`import deepeval.integrations.crewai` must not raise NameError when crewai
    is not installed; every public entry point raises a helpful ImportError.

    Previously the module crashed at import (``NameError: name
    'BaseEventListener'`` in handler.py, and undefined ``Crew``/``Agent``/``LLM``
    at the bottom of subs.py), and the ``is_crewai_installed()`` guard in subs.py
    was shadowed by a same-named flag, so it could never raise.
    """
    with dependency_absent("crewai", "deepeval.integrations.crewai"):
        ci = importlib.import_module("deepeval.integrations.crewai")
        for attr in ("Crew", "Agent", "LLM", "tool", "instrument_crewai"):
            assert hasattr(ci, attr)

        with pytest.raises(ImportError, match="CrewAI is not installed"):
            ci.Crew()
        with pytest.raises(ImportError, match="CrewAI is not installed"):
            ci.tool()
        with pytest.raises(ImportError, match="CrewAI is not installed"):
            ci.instrument_crewai()

        # The previously-shadowed guard must now actually raise.
        from deepeval.integrations.crewai.subs import is_crewai_installed

        with pytest.raises(ImportError, match="CrewAI is not installed"):
            is_crewai_installed()
