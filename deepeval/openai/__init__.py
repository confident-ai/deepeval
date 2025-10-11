try:
    import openai
except ImportError:
    raise ModuleNotFoundError("Please install OpenAI to use this feature: 'pip install openai'")

try:
    from openai import OpenAI, AsyncOpenAI  # noqa: F401
except ImportError:
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


if OpenAI or AsyncOpenAI:
    from deepeval.openai.patch import patch_openai_classes
    patch_openai_classes()