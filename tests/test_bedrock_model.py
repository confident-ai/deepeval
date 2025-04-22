"""Tests for Amazon Bedrock model implementations
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import base64
import json
from botocore.response import StreamingBody

from deepeval.models import BedrockModel, MultimodalBedrockModel
from deepeval.test_case import MLLMImage
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER

# Mock credentials for testing
TEST_REGION = "us-east-1"
TEST_RESPONSE = "This is a test response"
TEST_IMAGE_URL = "https://www.shutterstock.com/image-photo/funny-large-longhair-gray-kitten-600nw-1842198919.jpg"
TEST_LOCAL_IMAGE = "tests/data/test.jpg"

@pytest.fixture
def mock_anthropic_client():
    with patch('anthropic.AnthropicBedrock') as mock:
        message_mock = MagicMock()
        message_mock.content = [{"type": "text", "text": json.dumps({"response": TEST_RESPONSE})}]

        client_instance = MagicMock()
        client_instance.messages.create.return_value = message_mock

        mock.return_value = client_instance
        yield mock

@pytest.fixture
def mock_key_handler():
    with patch('deepeval.key_handler.KEY_FILE_HANDLER.fetch_data') as mock:
        mock.side_effect = lambda x: {
            KeyValues.AWS_REGION: TEST_REGION
        }.get(x)
        yield mock

class TestBedrockModel:
    """Test suite for Amazon Bedrock model"""

    def test_initialization(self):
        """Test model initialization with default parameters"""
        model = BedrockModel()

        assert model.model_id == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    def test_initialization_with_custom_params(self):
        """Test model initialization with custom parameters"""
        model = BedrockModel(
            model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            region=TEST_REGION
        )

        assert model.model_id == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert model.region == TEST_REGION

    def test_invalid_model_name(self):
        """Test initialization with invalid model name"""
        with pytest.raises(ValueError, match="Invalid model"):
            BedrockModel(model_id="invalid-model")

    @patch("deepeval.models.bedrock_model.AnthropicBedrock")
    def test_generate(self, mock_anthropic_client, mock_key_handler):
        """Test text generation"""
        mock_instance = mock_anthropic_client.return_value
        mock_instance.messages.create.return_value.content = [
            MagicMock(text=json.dumps({"response": TEST_RESPONSE}))
        ]

        model = BedrockModel()
        test_prompt = "Test prompt"
        response = model.generate(test_prompt)

        parsed = json.loads(response)
        assert isinstance(parsed, dict)
        assert "response" in parsed
        assert isinstance(parsed["response"], str)

        mock_instance.messages.create.assert_called_once()

    @patch("deepeval.models.bedrock_model.AsyncAnthropicBedrock")
    @pytest.mark.asyncio
    async def test_a_generate(self, mock_async_client, mock_key_handler):
        """Test async text generation"""
        message_mock = MagicMock()
        message_mock.content = [
            MagicMock(text=json.dumps({"response": TEST_RESPONSE}))
        ]

        mock_instance = MagicMock()
        mock_instance.messages.create = AsyncMock(return_value=message_mock)
        mock_async_client.return_value = mock_instance

        model = BedrockModel()
        response = await model.a_generate("Test prompt")

        parsed = json.loads(response)
        assert isinstance(parsed, dict)
        assert "response" in parsed
        assert isinstance(parsed["response"], str)

        mock_instance.messages.create.assert_called_once()


class TestBedrockMultimodalModel:
    """Test suite for Bedrock multimodal model (Anthropic Claude 3.7 Sonnet)."""

    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = MultimodalBedrockModel()

        assert model.model_id == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    def test_initialization_with_custom_params(self):
        """Test model initialization with custom parameters."""
        model = MultimodalBedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            region=TEST_REGION
        )

        assert model.model_id == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert model.region == TEST_REGION

    def test_invalid_model_name(self):
        """Test initialization with invalid model name."""
        with pytest.raises(ValueError, match="Invalid model"):
            MultimodalBedrockModel(model_id="invalid-model")

    def test_generate_prompt_local_image(mock_key_handler):
        """Test multimodal prompt generation with a local image."""
        model = MultimodalBedrockModel()

        multimodal_input = [
            "What's in these images?",
            MLLMImage(url=TEST_LOCAL_IMAGE, local=True)
        ]

        prompt = model.generate_prompt(multimodal_input)

        assert len(prompt) == 2

        text_block = prompt[0]["content"][0]
        assert text_block["type"] == "text"
        assert text_block["text"] == "What's in these images?"

        image_block = prompt[1]["content"][0]
        source = image_block["source"]
        assert image_block["type"] == "image"
        assert source["type"] == "base64"
        assert source["media_type"] == "image/jpeg"
        assert isinstance(source["data"], str)
        assert len(source["data"]) > 0


    def test_generate_prompt_remote_image(mock_key_handler):
        """Test multimodal prompt generation with a remote image."""
        model = MultimodalBedrockModel()

        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=TEST_IMAGE_URL, local=False)
        ]

        prompt = model.generate_prompt(multimodal_input)

        assert len(prompt) == 2

        text_block = prompt[0]["content"][0]
        assert text_block["type"] == "text"
        assert text_block["text"] == "Describe this image:"

        image_block = prompt[1]["content"][0]
        source = image_block["source"]
        assert image_block["type"] == "image"
        assert source["type"] == "base64"
        assert source["media_type"] == "image/jpeg"
        assert isinstance(source["data"], str)
        assert len(source["data"]) > 0


    @patch("deepeval.models.bedrock_model.AnthropicBedrock")
    def test_generate(self, mock_anthropic_client, mock_key_handler):
        """Test multimodal generation with image and text."""
        mock_instance = mock_anthropic_client.return_value
        mock_instance.messages.create.return_value.content = [
            MagicMock(text=json.dumps({"description": "A cat on a couch"}))
        ]

        model = MultimodalBedrockModel()

        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=TEST_LOCAL_IMAGE, local=True)
        ]

        response = model.generate(multimodal_input)
        parsed = json.loads(response)

        assert isinstance(parsed, dict)
        assert "description" in parsed
        assert isinstance(parsed["description"], str)

        mock_instance.messages.create.assert_called_once()


    @patch("deepeval.models.bedrock_model.AsyncAnthropicBedrock")
    @pytest.mark.asyncio
    async def test_a_generate(self, mock_async_client, mock_key_handler):
        """Test async multimodal generation."""
        message_mock = MagicMock()
        message_mock.content = [
            MagicMock(text=json.dumps({"description": "A dog on a beach"}))
        ]

        mock_instance = MagicMock()
        mock_instance.messages.create = AsyncMock(return_value=message_mock)
        mock_async_client.return_value = mock_instance

        model = MultimodalBedrockModel()

        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=TEST_IMAGE_URL, local=False)
        ]

        response = await model.a_generate(multimodal_input)
        parsed = json.loads(response)

        assert isinstance(parsed, dict)
        assert "description" in parsed
        assert isinstance(parsed["description"], str)

        mock_instance.messages.create.assert_called_once()

    def test_invalid_input_type(self, mock_key_handler):
        """Test handling of invalid input types."""
        model = MultimodalBedrockModel()

        multimodal_input = [
            "Describe this image:",
            {"url": TEST_IMAGE_URL}
        ]

        with pytest.raises(ValueError, match="Invalid input type"):
            model.generate_prompt(multimodal_input)
