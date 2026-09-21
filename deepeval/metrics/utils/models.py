from typing import Optional, Tuple, Union

from pydantic import SecretStr

from deepeval.config.mode import MODE_ENV_VAR, DeepEvalMode, is_experimental
from deepeval.config.settings import get_settings
from deepeval.errors import DeepEvalError
from deepeval.key_handler import (
    ModelKeyValues,
    EmbeddingKeyValues,
    KEY_FILE_HANDLER,
)
from deepeval.models import (
    DeepEvalBaseLLM,
    OpenAIModel,
    AnthropicModel,
    AzureOpenAIModel,
    OllamaModel,
    LocalModel,
    OpenAIEmbeddingModel,
    AzureOpenAIEmbeddingModel,
    OllamaEmbeddingModel,
    LocalEmbeddingModel,
    GeminiModel,
    AmazonBedrockModel,
    LiteLLMModel,
    PortkeyModel,
    KimiModel,
    GrokModel,
    DeepSeekModel,
    OpenRouterModel,
    TypeSafeModel,
)
from deepeval.models.base_model import (
    DeepEvalBaseEmbeddingModel,
    DeepEvalBaseSystemOneModel,
)
from deepeval.models.llms.constants import (
    OPENAI_MODELS_DATA,
    GEMINI_MODELS_DATA,
    OLLAMA_MODELS_DATA,
    ANTHROPIC_MODELS_DATA,
    GROK_MODELS_DATA,
    KIMI_MODELS_DATA,
)

MULTIMODAL_SUPPORTED_MODELS = {
    OpenAIModel: OPENAI_MODELS_DATA,
    GeminiModel: GEMINI_MODELS_DATA,
    OllamaModel: OLLAMA_MODELS_DATA,
    AzureOpenAIModel: OPENAI_MODELS_DATA,
    KimiModel: KIMI_MODELS_DATA,
    AnthropicModel: ANTHROPIC_MODELS_DATA,
    GrokModel: GROK_MODELS_DATA,
}

SETTINGS = get_settings()

###############################################
# Default Model Providers
###############################################


def should_use_anthropic_model():
    if SETTINGS.USE_ANTHROPIC_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_ANTHROPIC_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_azure_openai():
    if SETTINGS.USE_AZURE_OPENAI:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_AZURE_OPENAI)
    return value.lower() == "yes" if value is not None else False


def should_use_local_model():
    if SETTINGS.USE_LOCAL_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_LOCAL_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_ollama_model():
    local_key = SETTINGS.LOCAL_MODEL_API_KEY
    if local_key:
        # `LOCAL_MODEL_API_KEY` is a `SecretStr`, which never compares equal to
        # a plain string, so the sentinel is read out before comparing.
        if isinstance(local_key, SecretStr):
            local_key = local_key.get_secret_value()
        return local_key == "ollama"
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.LOCAL_MODEL_API_KEY)
    return value == "ollama"


def should_use_gemini_model():
    if SETTINGS.USE_GEMINI_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_GEMINI_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_openai_model():
    if SETTINGS.USE_OPENAI_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_OPENAI_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_litellm():
    if SETTINGS.USE_LITELLM:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_LITELLM)
    return value.lower() == "yes" if value is not None else False


def should_use_portkey():
    if SETTINGS.USE_PORTKEY_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_PORTKEY_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_deepseek_model():
    if SETTINGS.USE_DEEPSEEK_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_DEEPSEEK_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_openrouter_model():
    if SETTINGS.USE_OPENROUTER_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_OPENROUTER_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_moonshot_model():
    if SETTINGS.USE_MOONSHOT_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_MOONSHOT_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_grok_model():
    if SETTINGS.USE_GROK_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_GROK_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_amazon_bedrock_model():
    if SETTINGS.USE_AWS_BEDROCK_MODEL:
        return True
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_AWS_BEDROCK_MODEL)
    return value.lower() == "yes" if value is not None else False


###############################################
# LLM
###############################################


def initialize_model(
    model: Optional[Union[str, DeepEvalBaseLLM]] = None,
) -> Tuple[DeepEvalBaseLLM, bool]:
    """
    Returns a tuple of (initialized DeepEvalBaseLLM, using_native_model boolean)
    """
    # If model is natively supported, it should be deemed as using native model
    if is_native_model(model):
        return model, True
    # If model is a DeepEvalBaseLLM but not a native model, we can not assume it is a native model
    if isinstance(model, DeepEvalBaseLLM):
        return model, False
    if should_use_openai_model():
        return OpenAIModel(model=model), True
    if should_use_gemini_model():
        return GeminiModel(model=model), True
    if should_use_litellm():
        return LiteLLMModel(model=model), True
    if should_use_portkey():
        return PortkeyModel(model=model), True
    if should_use_ollama_model():
        return OllamaModel(model=model), True
    elif should_use_local_model():
        return LocalModel(model=model), True
    elif should_use_azure_openai():
        return AzureOpenAIModel(model=model), True
    elif should_use_moonshot_model():
        return KimiModel(model=model), True
    elif should_use_grok_model():
        return GrokModel(model=model), True
    elif should_use_deepseek_model():
        return DeepSeekModel(model=model), True
    elif should_use_openrouter_model():
        return OpenRouterModel(model=model), True
    elif should_use_anthropic_model():
        return AnthropicModel(model=model), True
    elif should_use_amazon_bedrock_model():
        return AmazonBedrockModel(model=model), True
    elif isinstance(model, str) or model is None:
        return OpenAIModel(model=model), True

    # Otherwise (the model is a wrong type), we raise an error
    raise TypeError(
        f"Unsupported type for model: {type(model)}. Expected None, str, DeepEvalBaseLLM, OpenAIModel, AzureOpenAIModel, LiteLLMModel, OllamaModel, LocalModel."
    )


def is_native_model(
    model: Optional[Union[str, DeepEvalBaseLLM]] = None,
) -> bool:
    if (
        isinstance(model, OpenAIModel)
        or isinstance(model, AnthropicModel)
        or isinstance(model, AzureOpenAIModel)
        or isinstance(model, OllamaModel)
        or isinstance(model, LocalModel)
        or isinstance(model, GeminiModel)
        or isinstance(model, AmazonBedrockModel)
        or isinstance(model, LiteLLMModel)
        or isinstance(model, KimiModel)
        or isinstance(model, GrokModel)
        or isinstance(model, DeepSeekModel)
        or isinstance(model, OpenRouterModel)
        or isinstance(model, PortkeyModel)
    ):
        return True
    else:
        return False


###############################################
# System One Model
###############################################


def initialize_system_one_model() -> Optional[DeepEvalBaseSystemOneModel]:
    """Jev answers QAG verdicts only under DEEPEVAL_MODE=experimental. There
    is no LLM fallback in that mode: a missing key or SDK is an error."""
    if not is_experimental():
        return None
    try:
        return TypeSafeModel()
    except DeepEvalError as e:
        raise DeepEvalError(
            f"{MODE_ENV_VAR}={DeepEvalMode.EXPERIMENTAL} routes metric "
            f"decisions to TypeSafe AI Jev, but it is not usable: {e} "
            f"Configure it with `deepeval set-typesafe --prompt-api-key` or "
            f"switch back with {MODE_ENV_VAR}={DeepEvalMode.STABLE}."
        ) from e


###############################################
# Multimodal Model
###############################################


###############################################
# Embedding Model
###############################################


def should_use_azure_openai_embedding():
    value = KEY_FILE_HANDLER.fetch_data(
        EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING
    )
    return value.lower() == "yes" if value is not None else False


def should_use_local_embedding():
    value = KEY_FILE_HANDLER.fetch_data(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)
    return value.lower() == "yes" if value is not None else False


def should_use_ollama_embedding():
    api_key = KEY_FILE_HANDLER.fetch_data(
        EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY
    )
    return api_key == "ollama"


def initialize_embedding_model(
    model: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None,
) -> DeepEvalBaseEmbeddingModel:
    if isinstance(model, DeepEvalBaseEmbeddingModel):
        return model
    if should_use_ollama_embedding():
        return OllamaEmbeddingModel()
    elif should_use_local_embedding():
        return LocalEmbeddingModel()
    elif should_use_azure_openai_embedding():
        return AzureOpenAIEmbeddingModel()
    elif isinstance(model, str) or model is None:
        return OpenAIEmbeddingModel(model=model)

    # Otherwise (the model is a wrong type), we raise an error
    raise TypeError(
        f"Unsupported type for embedding model: {type(model)}. Expected None, str, DeepEvalBaseEmbeddingModel, OpenAIEmbeddingModel, AzureOpenAIEmbeddingModel, OllamaEmbeddingModel, LocalEmbeddingModel."
    )
