"""File for handling API key"""

import json
from enum import Enum

from .constants import KEY_FILE, HIDDEN_DIR


class KeyValues(Enum):
    # Confident AI
    API_KEY = "api_key"
    CONFIDENT_REGION = "confident_region"
    # Cache
    LAST_TEST_RUN_LINK = "last_test_run_link"
    LAST_TEST_RUN_DATA = "last_test_run_data"
    # Azure Open AI
    AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
    OPENAI_API_VERSION = "OPENAI_API_VERSION"
    AZURE_DEPLOYMENT_NAME = "AZURE_DEPLOYMENT_NAME"
    AZURE_EMBEDDING_DEPLOYMENT_NAME = "AZURE_EMBEDDING_DEPLOYMENT_NAME"
    AZURE_MODEL_NAME = "AZURE_MODEL_NAME"
    AZURE_MODEL_VERSION = "AZURE_MODEL_VERSION"
    USE_AZURE_OPENAI = "USE_AZURE_OPENAI"
    USE_AZURE_OPENAI_EMBEDDING = "USE_AZURE_OPENAI_EMBEDDING"
    # Local Model
    LOCAL_MODEL_NAME = "LOCAL_MODEL_NAME"
    LOCAL_MODEL_BASE_URL = "LOCAL_MODEL_BASE_URL"
    LOCAL_MODEL_API_KEY = "LOCAL_MODEL_API_KEY"
    LOCAL_MODEL_FORMAT = "LOCAL_MODEL_FORMAT"
    USE_LOCAL_MODEL = "USE_LOCAL_MODEL"
    LOCAL_EMBEDDING_MODEL_NAME = "LOCAL_EMBEDDING_MODEL_NAME"
    LOCAL_EMBEDDING_BASE_URL = "LOCAL_EMBEDDING_BASE_URL"
    LOCAL_EMBEDDING_API_KEY = "LOCAL_EMBEDDING_API_KEY"
    USE_LOCAL_EMBEDDINGS = "USE_LOCAL_EMBEDDINGS"
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


class KeyFileHandler:
    def __init__(self):
        self.data = {}

    def write_key(self, key: KeyValues, value):
        """Appends or updates data in the hidden file"""
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
        with open(f"{HIDDEN_DIR}/{KEY_FILE}", "w") as f:
            json.dump(self.data, f)

    def fetch_data(self, key: KeyValues):
        """Fetches the data from the hidden file"""
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
        return self.data.get(key.value)

    def remove_key(self, key: KeyValues):
        """Removes the specified key from the data."""
        try:
            with open(f"{HIDDEN_DIR}/{KEY_FILE}", "r") as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    # Handle corrupted JSON file
                    self.data = {}
            self.data.pop(key.value, None)  # Remove the key if it exists
            with open(f"{HIDDEN_DIR}/{KEY_FILE}", "w") as f:
                json.dump(self.data, f)
        except FileNotFoundError:
            # Handle the case when the file doesn't exist
            pass  # No action needed if the file doesn't exist


KEY_FILE_HANDLER = KeyFileHandler()
