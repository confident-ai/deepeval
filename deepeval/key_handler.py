"""File for handling API key"""

import os
import json
import logging

from enum import Enum
from functools import lru_cache
from pydantic import SecretStr
from typing import get_args, get_origin, Union

from .constants import KEY_FILE, HIDDEN_DIR


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _secret_env_keys() -> frozenset[str]:
    # Lazy import avoids cycles at import time
    from deepeval.config.settings import Settings

    secret_keys: set[str] = set()
    for env_key, field in Settings.model_fields.items():
        ann = field.annotation
        if ann is SecretStr:
            secret_keys.add(env_key)
            continue

        origin = get_origin(ann)
        if origin is Union and any(a is SecretStr for a in get_args(ann)):
            secret_keys.add(env_key)

    return frozenset(secret_keys)


def _env_key_for_legacy_enum(key) -> str:
    # For ModelKeyValues, .name == .value, for KeyValues it's the important one:
    # KeyValues.API_KEY.name == "API_KEY" (matches Settings), value == "api_key" (legacy json key)
    return getattr(key, "name", str(key))


def _is_secret_key(key) -> bool:
    return _env_key_for_legacy_enum(key) in _secret_env_keys()


_WARNED_SECRET_KEYS = set()


class KeyValues(Enum):
    # Confident AI
    API_KEY = "api_key"
    CONFIDENT_API_KEY = "confident_api_key"
    CONFIDENT_BASE_URL = "confident_base_url"
    CONFIDENT_REGION = "confident_region"

    # Cache
    LAST_TEST_RUN_LINK = "last_test_run_link"
    LAST_TEST_RUN_DATA = "last_test_run_data"


class ModelKeyValues(Enum):
    # General
    TEMPERATURE = "TEMPERATURE"

    # Anthropic
    USE_ANTHROPIC_MODEL = "USE_ANTHROPIC_MODEL"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    ANTHROPIC_MODEL_NAME = "ANTHROPIC_MODEL_NAME"
    ANTHROPIC_COST_PER_INPUT_TOKEN = "ANTHROPIC_COST_PER_INPUT_TOKEN"
    ANTHROPIC_COST_PER_OUTPUT_TOKEN = "ANTHROPIC_COST_PER_OUTPUT_TOKEN"

    # AWS
    AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
    # AWS Bedrock
    USE_AWS_BEDROCK_MODEL = "USE_AWS_BEDROCK_MODEL"
    AWS_BEDROCK_MODEL_NAME = "AWS_BEDROCK_MODEL_NAME"
    AWS_BEDROCK_REGION = "AWS_BEDROCK_REGION"
    AWS_BEDROCK_COST_PER_INPUT_TOKEN = "AWS_BEDROCK_COST_PER_INPUT_TOKEN"
    AWS_BEDROCK_COST_PER_OUTPUT_TOKEN = "AWS_BEDROCK_COST_PER_OUTPUT_TOKEN"

    # Azure Open AI
    AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
    OPENAI_API_VERSION = "OPENAI_API_VERSION"
    AZURE_DEPLOYMENT_NAME = "AZURE_DEPLOYMENT_NAME"
    AZURE_MODEL_NAME = "AZURE_MODEL_NAME"
    AZURE_MODEL_VERSION = "AZURE_MODEL_VERSION"
    USE_AZURE_OPENAI = "USE_AZURE_OPENAI"

    # DeepSeek
    USE_DEEPSEEK_MODEL = "USE_DEEPSEEK_MODEL"
    DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
    DEEPSEEK_MODEL_NAME = "DEEPSEEK_MODEL_NAME"
    DEEPSEEK_COST_PER_INPUT_TOKEN = "DEEPSEEK_COST_PER_INPUT_TOKEN"
    DEEPSEEK_COST_PER_OUTPUT_TOKEN = "DEEPSEEK_COST_PER_OUTPUT_TOKEN"

    # Gemini
    USE_GEMINI_MODEL = "USE_GEMINI_MODEL"
    GOOGLE_API_KEY = "GOOGLE_API_KEY"
    GEMINI_MODEL_NAME = "GEMINI_MODEL_NAME"
    GOOGLE_GENAI_USE_VERTEXAI = "GOOGLE_GENAI_USE_VERTEXAI"
    GOOGLE_CLOUD_PROJECT = "GOOGLE_CLOUD_PROJECT"
    GOOGLE_CLOUD_LOCATION = "GOOGLE_CLOUD_LOCATION"
    GOOGLE_SERVICE_ACCOUNT_KEY = "GOOGLE_SERVICE_ACCOUNT_KEY"

    # Grok
    USE_GROK_MODEL = "USE_GROK_MODEL"
    GROK_API_KEY = "GROK_API_KEY"
    GROK_MODEL_NAME = "GROK_MODEL_NAME"
    GROK_COST_PER_INPUT_TOKEN = "GROK_COST_PER_INPUT_TOKEN"
    GROK_COST_PER_OUTPUT_TOKEN = "GROK_COST_PER_OUTPUT_TOKEN"

    # LiteLLM
    USE_LITELLM = "USE_LITELLM"
    LITELLM_API_KEY = "LITELLM_API_KEY"
    LITELLM_MODEL_NAME = "LITELLM_MODEL_NAME"
    LITELLM_API_BASE = "LITELLM_API_BASE"
    LITELLM_PROXY_API_BASE = "LITELLM_PROXY_API_BASE"
    LITELLM_PROXY_API_KEY = "LITELLM_PROXY_API_KEY"

    # LM Studio
    LM_STUDIO_API_KEY = "LM_STUDIO_API_KEY"
    LM_STUDIO_MODEL_NAME = "LM_STUDIO_MODEL_NAME"

    # Local Model
    USE_LOCAL_MODEL = "USE_LOCAL_MODEL"
    LOCAL_MODEL_API_KEY = "LOCAL_MODEL_API_KEY"
    LOCAL_MODEL_NAME = "LOCAL_MODEL_NAME"
    LOCAL_MODEL_BASE_URL = "LOCAL_MODEL_BASE_URL"
    LOCAL_MODEL_FORMAT = "LOCAL_MODEL_FORMAT"

    # Moonshot
    USE_MOONSHOT_MODEL = "USE_MOONSHOT_MODEL"
    MOONSHOT_API_KEY = "MOONSHOT_API_KEY"
    MOONSHOT_MODEL_NAME = "MOONSHOT_MODEL_NAME"
    MOONSHOT_COST_PER_INPUT_TOKEN = "MOONSHOT_COST_PER_INPUT_TOKEN"
    MOONSHOT_COST_PER_OUTPUT_TOKEN = "MOONSHOT_COST_PER_OUTPUT_TOKEN"

    # Ollama
    OLLAMA_MODEL_NAME = "OLLAMA_MODEL_NAME"

    # OpenAI
    USE_OPENAI_MODEL = "USE_OPENAI_MODEL"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_MODEL_NAME = "OPENAI_MODEL_NAME"
    OPENAI_COST_PER_INPUT_TOKEN = "OPENAI_COST_PER_INPUT_TOKEN"
    OPENAI_COST_PER_OUTPUT_TOKEN = "OPENAI_COST_PER_OUTPUT_TOKEN"

    # PortKey
    USE_PORTKEY_MODEL = "USE_PORTKEY_MODEL"
    PORTKEY_API_KEY = "PORTKEY_API_KEY"
    PORTKEY_MODEL_NAME = "PORTKEY_MODEL_NAME"
    PORTKEY_BASE_URL = "PORTKEY_BASE_URL"
    PORTKEY_PROVIDER_NAME = "PORTKEY_PROVIDER_NAME"

    # Vertex AI
    VERTEX_AI_MODEL_NAME = "VERTEX_AI_MODEL_NAME"

    # VLLM
    VLLM_API_KEY = "VLLM_API_KEY"
    VLLM_MODEL_NAME = "VLLM_MODEL_NAME"

    # OpenRouter
    USE_OPENROUTER_MODEL = "USE_OPENROUTER_MODEL"
    OPENROUTER_MODEL_NAME = "OPENROUTER_MODEL_NAME"
    OPENROUTER_COST_PER_INPUT_TOKEN = "OPENROUTER_COST_PER_INPUT_TOKEN"
    OPENROUTER_COST_PER_OUTPUT_TOKEN = "OPENROUTER_COST_PER_OUTPUT_TOKEN"
    OPENROUTER_API_KEY = "OPENROUTER_API_KEY"


class EmbeddingKeyValues(Enum):
    # Azure OpenAI
    USE_AZURE_OPENAI_EMBEDDING = "USE_AZURE_OPENAI_EMBEDDING"
    # Azure OpenAI
    AZURE_EMBEDDING_MODEL_NAME = "AZURE_EMBEDDING_MODEL_NAME"
    AZURE_EMBEDDING_DEPLOYMENT_NAME = "AZURE_EMBEDDING_DEPLOYMENT_NAME"

    # Local
    USE_LOCAL_EMBEDDINGS = "USE_LOCAL_EMBEDDINGS"
    LOCAL_EMBEDDING_MODEL_NAME = "LOCAL_EMBEDDING_MODEL_NAME"
    LOCAL_EMBEDDING_BASE_URL = "LOCAL_EMBEDDING_BASE_URL"
    LOCAL_EMBEDDING_API_KEY = ("LOCAL_EMBEDDING_API_KEY",)


class KeyFileHandler:
    def __init__(self):
        self.data = {}

    def _ensure_dir(self):
        os.makedirs(HIDDEN_DIR, exist_ok=True)

    def write_key(
        self, key: Union[KeyValues, ModelKeyValues, EmbeddingKeyValues], value
    ):
        """Appends or updates data in the hidden file"""

        # hard stop on secrets: never write to disk
        if _is_secret_key(key):
            logger.warning(
                "%s is a secret setting, refusing to persist. "
                "Keep your secrets in .env or .env.local instead.",
                _env_key_for_legacy_enum(key),
            )
            return

        try:
            with open(f"{HIDDEN_DIR}/{KEY_FILE}", "r") as f:
                # Load existing data
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    # Handle corrupted JSON file
                    self.data = {}
        except FileNotFoundError:
            # If file doesn't exist, start with an empty dictionary
            self.data = {}

        # Update the data with the new key-value pair
        self.data[key.value] = value

        # Write the updated data back to the file
        self._ensure_dir()
        with open(f"{HIDDEN_DIR}/{KEY_FILE}", "w") as f:
            json.dump(self.data, f)

    def fetch_data(
        self, key: Union[KeyValues, ModelKeyValues, EmbeddingKeyValues]
    ):
        """Fetches the data from the hidden file.
        NOTE: secrets in this file are deprecated; prefer env/.env."""
        try:
            with open(f"{HIDDEN_DIR}/{KEY_FILE}", "r") as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    # Handle corrupted JSON file
                    self.data = {}
        except FileNotFoundError:
            # Handle the case when the file doesn't exist
            self.data = {}

        value = self.data.get(key.value)

        # Deprecation: warn only if we're actually returning a secret
        if (
            value is not None
            and _is_secret_key(key)
            and _env_key_for_legacy_enum(key) not in _WARNED_SECRET_KEYS
        ):
            logger.warning(
                "Reading secret '%s' from legacy %s/%s. Persisting API keys in plaintext is deprecated. "
                "Move this to your environment (.env / .env.local). This fallback will be removed in a future release.",
                _env_key_for_legacy_enum(key),
                HIDDEN_DIR,
                KEY_FILE,
            )
            _WARNED_SECRET_KEYS.add(_env_key_for_legacy_enum(key))

        return value

    def remove_key(
        self, key: Union[KeyValues, ModelKeyValues, EmbeddingKeyValues]
    ):
        """Removes the specified key from the data."""
        try:
            with open(f"{HIDDEN_DIR}/{KEY_FILE}", "r") as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    # Handle corrupted JSON file
                    self.data = {}
            self.data.pop(key.value, None)  # Remove the key if it exists
            self._ensure_dir()
            with open(f"{HIDDEN_DIR}/{KEY_FILE}", "w") as f:
                json.dump(self.data, f)
        except FileNotFoundError:
            # Handle the case when the file doesn't exist
            pass  # No action needed if the file doesn't exist


KEY_FILE_HANDLER = KeyFileHandler()
