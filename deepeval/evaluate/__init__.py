import sys

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

_types_mod = __import__("types")


class _CallableModule(_types_mod.ModuleType):
    def __call__(self, *args, **kwargs):
        return evaluate(*args, **kwargs)


_mod = _CallableModule(__name__)
_mod.__dict__.update(
    {k: v for k, v in globals().items() if not k.startswith("_mod")}
)
_mod.__all__ = __all__
sys.modules[__name__] = _mod
