from deepeval.models.base_model import (
    DeepEvalBaseModel,
    DeepEvalBaseLLM,
    DeepEvalBaseEmbeddingModel,
    DeepEvalBaseTTS,
    DeepEvalBaseSTT,
)
from deepeval.models.llms import (
    OpenAIModel,
    AzureOpenAIModel,
    LocalModel,
    OllamaModel,
    AnthropicModel,
    GeminiModel,
    AmazonBedrockModel,
    LiteLLMModel,
    KimiModel,
    GrokModel,
    DeepSeekModel,
    PortkeyModel,
    OpenRouterModel,
)
from deepeval.models.embedding_models import (
    OpenAIEmbeddingModel,
    AzureOpenAIEmbeddingModel,
    LocalEmbeddingModel,
    OllamaEmbeddingModel,
)
from deepeval.models.tts import (
    CartesiaTTSModel,
    DeepgramTTSModel,
    ElevenLabsTTSModel,
    OpenAITTSModel,
)
from deepeval.models.stt import (
    AssemblyAISTTModel,
    CartesiaSTTModel,
    DeepgramSTTModel,
    ElevenLabsSTTModel,
    OpenAISTTModel,
)

__all__ = [
    "DeepEvalBaseModel",
    "DeepEvalBaseLLM",
    "DeepEvalBaseEmbeddingModel",
    "DeepEvalBaseTTS",
    "DeepEvalBaseSTT",
    "OpenAIModel",
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
    "OpenAIEmbeddingModel",
    "AzureOpenAIEmbeddingModel",
    "LocalEmbeddingModel",
    "OllamaEmbeddingModel",
    "PortkeyModel",
    "OpenRouterModel",
    "OpenAITTSModel",
    "OpenAISTTModel",
    "CartesiaTTSModel",
    "DeepgramTTSModel",
    "ElevenLabsTTSModel",
    "AssemblyAISTTModel",
    "CartesiaSTTModel",
    "DeepgramSTTModel",
    "ElevenLabsSTTModel",
]


def __getattr__(name: str):
    if name == "GPTModel":
        from deepeval.models.llms.openai_model import warn_gpt_model_deprecated

        warn_gpt_model_deprecated()
        return OpenAIModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
