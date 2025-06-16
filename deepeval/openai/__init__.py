import os
import sys
import importlib.util
from importlib.machinery import SourceFileLoader

from deepeval.openai.patch import patch_openai

# ——— 1) Locate the real OpenAI package on disk ———
spec = importlib.util.find_spec("openai")
if not spec or not spec.origin:
    raise ImportError("OpenAI package not found")
origin = spec.origin
subpkg_paths = spec.submodule_search_locations

# ——— 2) Prepare a “fork” module loader ———
loader = SourceFileLoader("deepeval_openai_fork", origin)
fork_spec = importlib.util.spec_from_loader(
    "deepeval_openai_fork", loader, origin=origin, is_package=True
)

# ——— 3) Instantiate the module and preserve its package path ———
fork = importlib.util.module_from_spec(fork_spec)
fork.__path__ = subpkg_paths  # so internal imports still resolve

# ——— 4) Register, load, and patch the forked copy ———
sys.modules[fork_spec.name] = fork
loader.exec_module(fork)
patch_openai(fork)

# ——— 5) Export only inside deepeval.openai ———
openai      = fork
OpenAI      = fork.OpenAI
AsyncOpenAI = fork.AsyncOpenAI

__all__ = ["openai", "OpenAI", "AsyncOpenAI"] + [
    name for name in dir(openai) if not name.startswith("_")
]
