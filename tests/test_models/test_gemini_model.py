"""Tests for Gemini model implementations"""

import pytest
from unittest.mock import patch, MagicMock
from google.genai import types

from deepeval.models import GeminiModel, MultimodalGeminiModel
from deepeval.test_case import MLLMImage
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER

# Mock credentials for testing
TEST_API_KEY = ""
TEST_PROJECT = "test-project"
TEST_LOCATION = "us-central1"
TEST_RESPONSE = "This is a test response"

safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]


@pytest.fixture
def mock_client():
    with patch("google.genai.Client") as mock:
        instance = mock.return_value

        # Mock synchronous models
        sync_models = MagicMock()
        sync_models.generate_content.return_value = MagicMock(
            text=TEST_RESPONSE
        )

        # Mock asynchronous models
        async_models = MagicMock()
        async_response = MagicMock(text=TEST_RESPONSE)

        # Create an async function to return the response
        async def async_magic():
            return async_response

        # Set the async generate_content to return our async function
        async_models.generate_content.return_value = async_magic()

        # Assign the mock models to the client instance
        instance.models = sync_models
        instance.aio.models = async_models

        yield mock


@pytest.fixture
def mock_key_handler():
    with patch("deepeval.key_handler.KEY_FILE_HANDLER.fetch_data") as mock:
        mock.side_effect = lambda x: {
            KeyValues.GOOGLE_API_KEY: TEST_API_KEY,
            KeyValues.GOOGLE_CLOUD_PROJECT: TEST_PROJECT,
            KeyValues.GOOGLE_CLOUD_LOCATION: TEST_LOCATION,
            KeyValues.GOOGLE_GENAI_USE_VERTEXAI: "true",
        }.get(x)
        yield mock


class TestGeminiModel:
    """Test suite for text-only Gemini model"""

    def test_initialization(self, mock_client, mock_key_handler):
        """Test model initialization with default parameters"""
        model = GeminiModel()

        # Verify model initialization
        assert model.model_name == "gemini-1.5-pro"
        assert model.project == TEST_PROJECT
        assert model.location == TEST_LOCATION
        assert model.get_model_name() == "gemini-1.5-pro"

        # Verify Client initialization
        mock_client.assert_called_once_with(
            vertexai=True, project=TEST_PROJECT, location=TEST_LOCATION
        )

    def test_initialization_with_custom_params(
        self, mock_client, mock_key_handler
    ):
        """Test model initialization with custom parameters"""
        model = GeminiModel(
            model_name="gemini-1.5-flash",
            project="custom-project",
            location="europe-west4",
        )

        assert model.model_name == "gemini-1.5-flash"
        assert model.project == "custom-project"
        assert model.location == "europe-west4"
        assert model.get_model_name() == "gemini-1.5-flash"

    def test_invalid_model_name(self):
        """Test initialization with invalid model name"""
        with pytest.raises(ValueError, match="Invalid model"):
            GeminiModel(model_name="invalid-model")

    def test_generate(self, mock_client, mock_key_handler):
        """Test text generation"""
        model = GeminiModel()
        response = model.generate("Test prompt")

        # Verify response
        assert response == TEST_RESPONSE

        # Verify generate_content was called with correct parameters
        mock_client.return_value.models.generate_content.assert_called_once_with(
            model=model.model_name,
            contents="Test prompt",
            config=types.GenerateContentConfig(
                safety_settings=safety_settings, temperature=0.0
            ),
        )

    @pytest.mark.asyncio
    async def test_a_generate(self, mock_client, mock_key_handler):
        """Test async text generation"""
        model = GeminiModel()
        response = await model.a_generate("Test prompt")

        # Verify response
        assert response == TEST_RESPONSE

        # Verify generate_content_async was called with correct parameters
        mock_client.return_value.aio.models.generate_content.assert_called_once_with(
            model=model.model_name,
            contents="Test prompt",
            config=types.GenerateContentConfig(
                safety_settings=safety_settings, temperature=0.0
            ),
        )


class TestMultimodalGeminiModel:
    """Test suite for multimodal Gemini model"""

    def test_initialization(self, mock_client, mock_key_handler):
        """Test model initialization with default parameters"""
        model = MultimodalGeminiModel()

        # Verify model initialization
        assert model.model_name == "gemini-1.5-pro"
        assert model.project == TEST_PROJECT
        assert model.location == TEST_LOCATION
        assert model.get_model_name() == "gemini-1.5-pro"

        # Verify Client initialization
        mock_client.assert_called_once_with(
            vertexai=True, project=TEST_PROJECT, location=TEST_LOCATION
        )

    def test_initialization_with_custom_params(
        self, mock_client, mock_key_handler
    ):
        """Test model initialization with custom parameters"""
        model = MultimodalGeminiModel(
            model_name="gemini-1.5-flash",
            project="custom-project",
            location="europe-west4",
        )

        assert model.model_name == "gemini-1.5-flash"
        assert model.project == "custom-project"
        assert model.location == "europe-west4"
        assert model.get_model_name() == "gemini-1.5-flash"

    def test_invalid_model_name(self):
        """Test initialization with invalid model name"""
        with pytest.raises(ValueError, match="Invalid model"):
            MultimodalGeminiModel(model_name="invalid-model")

    def test_generate(self, mock_client, mock_key_handler):
        """Test multimodal generation"""
        model = MultimodalGeminiModel()

        # Create test input
        test_url = "https://example.com/test.jpg"
        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=test_url, local=False),
        ]

        # Get the expected prompt that will be passed to generate_content
        prompt = model.generate_prompt(multimodal_input)
        response = model.generate(multimodal_input)

        # Verify response
        assert response == TEST_RESPONSE

        # Verify mock calls
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.assert_called_once_with(
            model=model.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=safety_settings, temperature=0.0
            ),
        )

    @pytest.mark.asyncio
    async def test_a_generate(self, mock_client, mock_key_handler):
        """Test async multimodal generation"""
        model = MultimodalGeminiModel()

        # Create test input
        test_url = "https://example.com/test.jpg"
        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=test_url, local=False),
        ]

        # Get the expected prompt that will be passed to generate_content
        prompt = model.generate_prompt(multimodal_input)
        response = await model.a_generate(multimodal_input)

        # Verify response
        assert response == TEST_RESPONSE

        # Verify mock calls
        mock_instance = mock_client.return_value
        mock_instance.aio.models.generate_content.assert_called_once_with(
            model=model.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=safety_settings, temperature=0.0
            ),
        )

    def test_invalid_input_type(self, mock_client, mock_key_handler):
        """Test handling of invalid input types"""
        model = MultimodalGeminiModel()

        # Create test input with invalid type
        multimodal_input = [
            "Describe this image:",
            {"url": "test.jpg"},  # Invalid type
        ]

        with pytest.raises(ValueError, match="Invalid input type"):
            model.generate_prompt(multimodal_input)
