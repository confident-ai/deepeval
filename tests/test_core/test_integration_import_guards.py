"""
Regression tests: optional integration modules must import cleanly even when
their optional third-party dependency is missing, and only raise a *helpful*
ImportError when actually used.

The dependency's absence is simulated via an import blocker so these run
deterministically regardless of whether the dependency happens to be installed
in the test environment.
"""

import sys
import importlib

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


def test_langchain_integration_imports_without_langchain():
    """`import deepeval.integrations.langchain` must not raise NameError when
    langchain is not installed; using the integration raises a helpful
    ImportError instead.

    Previously the module referenced ``BaseCallbackHandler`` as a base class at
    class-definition time, so importing without langchain crashed with
    ``NameError: name 'BaseCallbackHandler' is not defined`` — which made the
    ``is_langchain_installed()`` guard unreachable dead code.
    """
    module_name = "deepeval.integrations.langchain"
    watch = ("langchain", module_name)

    # Snapshot and clear anything cached so the import runs fresh under the block.
    saved = {k: v for k, v in sys.modules.items() if k.startswith(watch)}
    for key in list(sys.modules):
        if key.startswith(watch):
            del sys.modules[key]

    finder = _BlockModules("langchain")
    sys.meta_path.insert(0, finder)
    try:
        lc = importlib.import_module(module_name)
        assert hasattr(lc, "CallbackHandler")
        assert hasattr(lc, "tool")

        # Actually using the integration surfaces the intended, helpful error.
        with pytest.raises(ImportError, match="LangChain is not installed"):
            lc.CallbackHandler()
        with pytest.raises(ImportError, match="LangChain is not installed"):
            lc.tool()
    finally:
        sys.meta_path.remove(finder)
        for key in list(sys.modules):
            if key.startswith(watch):
                del sys.modules[key]
        sys.modules.update(saved)
