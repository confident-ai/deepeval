from typing import Optional, Tuple, Union

from pydantic import SecretStr

from deepeval.config.eval_mode import (
    EVAL_MODE_ENV_VAR,
    EvalMode,
    EvalModeName,
    resolve_eval_mode,
)
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
    eval_mode: Optional[Union[EvalModeName, EvalMode]] = None,
) -> Tuple[Optional[DeepEvalBaseLLM], bool]:
    """
    Returns a tuple of (initialized DeepEvalBaseLLM, using_native_model boolean)

    ``eval_mode`` is the calling metric's eval mode. Under ``system_one`` Jev
    runs the whole metric and nothing falls back to the LLM, so no LLM is
    built (and a missing LLM key is not an error): the result is
    ``(None, True)``. ``True`` keeps the metric's cost tracking on, which Jev
    reports into. Leave ``eval_mode`` as ``None`` when the caller always
    needs the LLM (synthesizer, simulator, metrics with no System One form).
    """
    if eval_mode is not None and _as_mode(eval_mode) is EvalMode.SYSTEM_ONE:
        return None, True
    return _build_model(model)


def _as_mode(eval_mode: Union[EvalModeName, EvalMode]) -> EvalMode:
    if isinstance(eval_mode, EvalMode):
        return eval_mode
    return resolve_eval_mode(eval_mode)


def _build_model(
    model: Optional[Union[str, DeepEvalBaseLLM]] = None,
) -> Tuple[DeepEvalBaseLLM, bool]:
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


def initialize_system_one_model(
    model: Optional[Union[str, DeepEvalBaseSystemOneModel]] = None,
    eval_mode: Optional[Union[EvalModeName, EvalMode]] = None,
) -> Optional[DeepEvalBaseSystemOneModel]:
    """Build the System One model a metric decides with, or ``None`` when its
    eval mode never calls one.

    ``model`` is the metric's ``system_one_model`` argument: a configured
    ``DeepEvalBaseSystemOneModel``, a TypeSafe model name, or ``None`` for
    the default. ``eval_mode`` is the metric's eval mode; ``None`` resolves
    it from settings (for metrics that take no ``eval_mode`` argument).
    Under ``llm`` no model is built, so a missing TypeSafe key is not an
    error; under ``hybrid`` and ``system_one`` a missing key or SDK fails
    here, at construction, rather than mid-evaluation."""
    mode = _as_mode(eval_mode) if eval_mode is not None else resolve_eval_mode()
    if not mode.uses_system_one:
        return None
    if isinstance(model, DeepEvalBaseSystemOneModel):
        return model
    if model is not None and not isinstance(model, str):
        raise TypeError(
            f"Unsupported type for system_one_model: {type(model)}. Expected "
            "None, str, or DeepEvalBaseSystemOneModel."
        )
    try:
        return TypeSafeModel(model=model)
    except DeepEvalError as e:
        raise DeepEvalError(
            f"{EVAL_MODE_ENV_VAR}={mode} routes metric decisions to TypeSafe "
            f"AI Jev, but it is not usable: {e} Configure it with `deepeval "
            f"set-typesafe --prompt-api-key` or switch back with `deepeval "
            f"set-eval-mode {EvalMode.LLM}`."
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
