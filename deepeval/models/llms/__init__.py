from .azure_model import AzureOpenAIModel
from .openai_model import OpenAIModel, warn_gpt_model_deprecated
from .local_model import LocalModel
from .ollama_model import OllamaModel
from .gemini_model import GeminiModel
from .anthropic_model import AnthropicModel
from .amazon_bedrock_model import AmazonBedrockModel
from .litellm_model import LiteLLMModel
from .kimi_model import KimiModel
from .grok_model import GrokModel
from .deepseek_model import DeepSeekModel
from .portkey_model import PortkeyModel
from .openrouter_model import OpenRouterModel

__all__ = [
    "AzureOpenAIModel",
    "OpenAIModel",
    "LocalModel",
    "OllamaModel",
    "GeminiModel",
    "AnthropicModel",
    "AmazonBedrockModel",
    "LiteLLMModel",
    "KimiModel",
    "GrokModel",
    "DeepSeekModel",
    "PortkeyModel",
    "OpenRouterModel",
]


def __getattr__(name: str):
    if name == "GPTModel":
        warn_gpt_model_deprecated()
        return OpenAIModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
