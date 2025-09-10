"""File for handling API key"""

import os
import json
import logging

from enum import Enum
from typing import Union

from .constants import KEY_FILE, HIDDEN_DIR


logger = logging.getLogger(__name__)


SECRET_KEYS = {
    # General providers
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    # Azure OpenAI
    "AZURE_OPENAI_API_KEY",
    # Google / Gemini
    "GOOGLE_API_KEY",
    # xAI Grok
    "GROK_API_KEY",
    # Moonshot
    "MOONSHOT_API_KEY",
    # DeepSeek
    "DEEPSEEK_API_KEY",
    # LiteLLM
    "LITELLM_API_KEY",
    # Local gateways (if any require keys)
    "LOCAL_MODEL_API_KEY",
    "LOCAL_EMBEDDING_API_KEY",
}

_WARNED_SECRET_KEYS = set()


class KeyValues(Enum):
    # Confident AI
    API_KEY = "api_key"
    CONFIDENT_REGION = "confident_region"
    # Cache
    LAST_TEST_RUN_LINK = "last_test_run_link"
    LAST_TEST_RUN_DATA = "last_test_run_data"


class ModelKeyValues(Enum):
    # General
    TEMPERATURE = "TEMPERATURE"
    # Azure Open AI
    AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
    OPENAI_API_VERSION = "OPENAI_API_VERSION"
    AZURE_DEPLOYMENT_NAME = "AZURE_DEPLOYMENT_NAME"
    AZURE_MODEL_NAME = "AZURE_MODEL_NAME"
    AZURE_MODEL_VERSION = "AZURE_MODEL_VERSION"
    USE_AZURE_OPENAI = "USE_AZURE_OPENAI"
    # Local Model
    LOCAL_MODEL_NAME = "LOCAL_MODEL_NAME"
    LOCAL_MODEL_BASE_URL = "LOCAL_MODEL_BASE_URL"
    LOCAL_MODEL_API_KEY = "LOCAL_MODEL_API_KEY"
    LOCAL_MODEL_FORMAT = "LOCAL_MODEL_FORMAT"
    USE_LOCAL_MODEL = "USE_LOCAL_MODEL"
    # Gemini
    USE_GEMINI_MODEL = "USE_GEMINI_MODEL"
    GEMINI_MODEL_NAME = "GEMINI_MODEL_NAME"
    GOOGLE_API_KEY = "GOOGLE_API_KEY"
    GOOGLE_GENAI_USE_VERTEXAI = "GOOGLE_GENAI_USE_VERTEXAI"
    GOOGLE_CLOUD_PROJECT = "GOOGLE_CLOUD_PROJECT"
    GOOGLE_CLOUD_LOCATION = "GOOGLE_CLOUD_LOCATION"
    # LiteLLM
    USE_LITELLM = "USE_LITELLM"
    LITELLM_MODEL_NAME = "LITELLM_MODEL_NAME"
    LITELLM_API_KEY = "LITELLM_API_KEY"
    LITELLM_API_BASE = "LITELLM_API_BASE"
    # OpenAI
    USE_OPENAI_MODEL = "USE_OPENAI_MODEL"
    OPENAI_MODEL_NAME = "OPENAI_MODEL_NAME"
    OPENAI_COST_PER_INPUT_TOKEN = "OPENAI_COST_PER_INPUT_TOKEN"
    OPENAI_COST_PER_OUTPUT_TOKEN = "OPENAI_COST_PER_OUTPUT_TOKEN"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    # Moonshot
    USE_MOONSHOT_MODEL = "USE_MOONSHOT_MODEL"
    MOONSHOT_MODEL_NAME = "MOONSHOT_MODEL_NAME"
    MOONSHOT_API_KEY = "MOONSHOT_API_KEY"
    # Grok
    USE_GROK_MODEL = "USE_GROK_MODEL"
    GROK_MODEL_NAME = "GROK_MODEL_NAME"
    GROK_API_KEY = "GROK_API_KEY"
    # DeepSeek
    USE_DEEPSEEK_MODEL = "USE_DEEPSEEK_MODEL"
    DEEPSEEK_MODEL_NAME = "DEEPSEEK_MODEL_NAME"
    DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"


class EmbeddingKeyValues(Enum):
    # Azure OpenAI
    USE_AZURE_OPENAI_EMBEDDING = "USE_AZURE_OPENAI_EMBEDDING"
    AZURE_EMBEDDING_DEPLOYMENT_NAME = "AZURE_EMBEDDING_DEPLOYMENT_NAME"
    # Local
    USE_LOCAL_EMBEDDINGS = "USE_LOCAL_EMBEDDINGS"
    LOCAL_EMBEDDING_MODEL_NAME = "LOCAL_EMBEDDING_MODEL_NAME"
    LOCAL_EMBEDDING_BASE_URL = "LOCAL_EMBEDDING_BASE_URL"
    LOCAL_EMBEDDING_API_KEY = "LOCAL_EMBEDDING_API_KEY"


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
        if key.value in SECRET_KEYS:
            logger.warning(
                f"{key} is blacklisted, refusing to persist. Keep your secrets in .env or .env.local instead"
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
            and key.value in SECRET_KEYS
            and key.value not in _WARNED_SECRET_KEYS
        ):
            logger.warning(
                f"Reading secret '{key.value}' from legacy {HIDDEN_DIR}/{KEY_FILE}. "
                "Persisting API keys in plaintext is deprecated. "
                "Move this to your environment (.env / .env.local). "
                "This fallback will be removed in a future release."
            )
            _WARNED_SECRET_KEYS.add(key.value)

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
