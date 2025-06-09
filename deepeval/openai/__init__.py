import openai as _openai
from deepeval.openai.patch import patch_openai

patch_openai(_openai)
openai = _openai
OpenAI      = _openai.OpenAI
AsyncOpenAI = _openai.AsyncOpenAI

__all__ = [
    "openai",
    "OpenAI",
    "AsyncOpenAI"
] + [
    name for name in dir(_openai) 
    if not name.startswith("_")
]