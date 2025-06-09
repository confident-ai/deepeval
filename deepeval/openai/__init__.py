# import the real OpenAI SDK
import openai as _openai

# apply your tracing/monkey-patch logic
from deepeval.openai.patch import patch_openai
patch_openai(_openai)

# alias the module itself so users can import it by name
openai = _openai
OpenAI      = _openai.OpenAI
AsyncOpenAI = _openai.AsyncOpenAI

# re-export everything, including our new `openai` symbol
__all__ = [
    "openai",
    "OpenAI",
    "AsyncOpenAI"
] + [
    name for name in dir(_openai) 
    if not name.startswith("_")
]