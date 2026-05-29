import sys
import importlib

_stdlib_types = importlib.import_module("types")

from .evaluate import evaluate, assert_test
from .compare import compare
from .configs import AsyncConfig, DisplayConfig, CacheConfig, ErrorConfig

__all__ = [
    "evaluate",
    "assert_test",
    "compare",
    "AsyncConfig",
    "DisplayConfig",
    "CacheConfig",
    "ErrorConfig",
]


class _CallableEvaluateModule(_stdlib_types.ModuleType):
    """Module subclass that lets ``deepeval.evaluate(...)`` be called directly.

    Without this, ``deepeval/__init__.py`` had to shadow the subpackage with the
    bare function, which broke dotted submodule access such as
    ``import deepeval.evaluate.configs; deepeval.evaluate.configs.AsyncConfig()``.
    """

    def __call__(self, *args, **kwargs):
        from .evaluate import evaluate as _fn

        return _fn(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableEvaluateModule
