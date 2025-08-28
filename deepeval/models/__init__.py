import importlib
from typing import TYPE_CHECKING

__all__ = [
    "DeepEvalBaseModel",
    "DeepEvalBaseLLM",
    "DeepEvalBaseMLLM",
    "DeepEvalBaseEmbeddingModel",
    "GPTModel",
    "AzureOpenAIModel",
    "LocalModel",
    "OllamaModel",
    "AnthropicModel",
    "GeminiModel",
    "AmazonBedrockModel",
    "LiteLLMModel",
    "KimiModel",
    "GrokModel",
    "DeepSeekModel",
    "MultimodalOpenAIModel",
    "MultimodalOllamaModel",
    "MultimodalGeminiModel",
    "OpenAIEmbeddingModel",
    "AzureOpenAIEmbeddingModel",
    "LocalEmbeddingModel",
    "OllamaEmbeddingModel",
]

_SYMBOL_TO_MODULE = {
    # Base types
    "DeepEvalBaseModel": "deepeval.models.base_model",
    "DeepEvalBaseLLM": "deepeval.models.base_model",
    "DeepEvalBaseMLLM": "deepeval.models.base_model",
    "DeepEvalBaseEmbeddingModel": "deepeval.models.base_model",
    # LLMs
    "GPTModel": "deepeval.models.llms.openai_model",
    "AzureOpenAIModel": "deepeval.models.llms.azure_model",
    "LocalModel": "deepeval.models.llms.local_model",
    "OllamaModel": "deepeval.models.llms.ollama_model",
    "AnthropicModel": "deepeval.models.llms.anthropic_model",
    "GeminiModel": "deepeval.models.llms.gemini_model",
    "AmazonBedrockModel": "deepeval.models.llms.amazon_bedrock_model",
    "LiteLLMModel": "deepeval.models.llms.litellm_model",
    "KimiModel": "deepeval.models.llms.kimi_model",
    "GrokModel": "deepeval.models.llms.grok_model",
    "DeepSeekModel": "deepeval.models.llms.deepseek_model",
    # Multimodal LLMs
    "MultimodalOpenAIModel": "deepeval.models.mlllms.openai_model",
    "MultimodalOllamaModel": "deepeval.models.mlllms.ollama_model",
    "MultimodalGeminiModel": "deepeval.models.mlllms.gemini_model",
    # Embedding models
    "OpenAIEmbeddingModel": "deepeval.models.embedding_models.openai_embedding_model",
    "AzureOpenAIEmbeddingModel": "deepeval.models.embedding_models.azure_embedding_model",
    "LocalEmbeddingModel": "deepeval.models.embedding_models.local_embedding_model",
    "OllamaEmbeddingModel": "deepeval.models.embedding_models.ollama_embedding_model",
}


def __getattr__(name: str):
    module_path = _SYMBOL_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(name)
    module = importlib.import_module(module_path)
    obj = getattr(module, name)
    globals()[name] = obj  # cache for subsequent access
    return obj


if TYPE_CHECKING:
    from deepeval.models.base_model import (
        DeepEvalBaseEmbeddingModel,
        DeepEvalBaseLLM,
        DeepEvalBaseMLLM,
        DeepEvalBaseModel,
    )
    from deepeval.models.embedding_models import (
        AzureOpenAIEmbeddingModel,
        LocalEmbeddingModel,
        OllamaEmbeddingModel,
        OpenAIEmbeddingModel,
    )
    from deepeval.models.llms import (
        AmazonBedrockModel,
        AnthropicModel,
        AzureOpenAIModel,
        DeepSeekModel,
        GeminiModel,
        GPTModel,
        GrokModel,
        KimiModel,
        LiteLLMModel,
        LocalModel,
        OllamaModel,
    )
    from deepeval.models.mlllms import (
        MultimodalGeminiModel,
        MultimodalOllamaModel,
        MultimodalOpenAIModel,
    )
