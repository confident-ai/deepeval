from importlib.machinery import SourceFileLoader
import importlib.util
import sys

from deepeval.openai.patch import patch_openai


def load_and_patch_openai():
    openai_spec = importlib.util.find_spec("openai")
    if not openai_spec or not openai_spec.origin:
        raise ImportError("Could not find the OpenAI package")
    package_dirs = openai_spec.submodule_search_locations
    loader = SourceFileLoader("deepeval_openai", openai_spec.origin)
    new_spec = importlib.util.spec_from_loader(
        "deepeval_openai",
        loader,
        origin=openai_spec.origin,
        is_package=True,
    )
    deepeval_openai = importlib.util.module_from_spec(new_spec)
    deepeval_openai.__path__ = package_dirs
    sys.modules["deepeval_openai"] = deepeval_openai
    loader.exec_module(deepeval_openai)
    patch_openai(deepeval_openai)
    return deepeval_openai


patched_openai = load_and_patch_openai()
openai = patched_openai
OpenAI = patched_openai.OpenAI
AsyncOpenAI = patched_openai.AsyncOpenAI

__all__ = [
    "openai",
    "OpenAI",
    "AsyncOpenAI",
]
