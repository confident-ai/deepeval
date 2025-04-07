"""Tests for Amazon Bedrock model implementations
"""

import pytest
from unittest.mock import patch, MagicMock
import base64
from botocore.response import StreamingBody

from deepeval.models import BedrockModel, MultimodalBedrockModel
from deepeval.test_case import MLLMImage
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER

# Mock credentials for testing
TEST_REGION = "us-east-1"
TEST_RESPONSE_JSON = '{"content": [{"type": "text", "text": "This is a test response"}]}'
TEST_RESPONSE = "This is a test response"
TEST_IMAGE_URL = "https://www.shutterstock.com/image-photo/funny-large-longhair-gray-kitten-600nw-1842198919.jpg"
TEST_LOCAL_IMAGE = "tests/data/test.jpg"

@pytest.fixture
def mock_anthropic_client():
    with patch('anthropic.AnthropicBedrock') as mock:
        client_instance = MagicMock()
        client_instance.messages.create.return_value = MagicMock(
            content=[{"type": "text", "text": "This is a test response"}]
        )
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

    def test_initialization(self, mock_boto3_client, mock_key_handler):
        """Test model initialization with default parameters"""
        model = BedrockModel()

        assert model.model_id == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert model.region == TEST_REGION

        mock_boto3_client.assert_called_once_with('bedrock-runtime', region_name='us-east-1', aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None)

    def test_initialization_with_custom_params(self, mock_boto3_client):
        """Test model initialization with custom parameters"""
        model = BedrockModel(
            model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            region="us-west-2"
        )

        assert model.model_id == "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        assert model.region == "us-west-2"

    def test_invalid_model_name(self):
        """Test initialization with invalid model name"""
        with pytest.raises(ValueError, match="Invalid model"):
            BedrockModel(model_id="invalid-model")

    def test_generate(self, mock_anthropic_client, mock_key_handler):
        """Test text generation"""
        model = BedrockModel()
        test_prompt = "Test prompt"
        response = model.generate(test_prompt)

        assert response == TEST_RESPONSE

        mock_instance = mock_anthropic_client.return_value
        mock_instance.invoke_model.assert_called_once()

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropicBedrock")
    async def test_a_generate(self, mock_async_client, mock_key_handler):
        mock_instance = MagicMock()
        mock_instance.messages.create.return_value = MagicMock(
            content=[{"type": "text", "text": "This is a test response"}]
        )
        mock_async_client.return_value = mock_instance

        model = BedrockModel()
        test_prompt = "Test prompt"
        response = await model.a_generate(test_prompt)

        assert response == TEST_RESPONSE
        mock_instance.messages.create.assert_called_once()


class TestBedrockMultimodalModel:
    """Test suite for Bedrock multimodal model (Anthropic Claude 3.7 Sonnet)."""

    def test_initialization(self, mock_boto3_client, mock_key_handler):
        """Test model initialization with default parameters."""
        model = MultimodalBedrockModel()

        assert model.model_id == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert model.region == TEST_REGION

        mock_boto3_client.assert_called_once_with('bedrock-runtime', region_name='us-east-1', aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None)

    def test_initialization_with_custom_params(self, mock_boto3_client):
        """Test model initialization with custom parameters."""
        model = MultimodalBedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            region="us-west-2"
        )

        assert model.model_id == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert model.region == "us-west-2"

    def test_invalid_model_name(self):
        """Test initialization with invalid model name."""
        with pytest.raises(ValueError, match="Invalid model"):
            MultimodalBedrockModel(model_id="invalid-model")


    def test_generate_prompt_local_image(self, mock_boto3_client, mock_key_handler):
        """Test multimodal prompt generation with a local image."""
        model = MultimodalBedrockModel()

        with open(TEST_LOCAL_IMAGE, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        multimodal_input = [
            "What's in these images?",
            MLLMImage(url=TEST_LOCAL_IMAGE, local=True)
        ]
        
        prompt = model.generate_prompt(multimodal_input)

        assert isinstance(prompt, list)
        assert len(prompt) == 2

        print(f"Generated Prompt: {prompt}")

        assert isinstance(prompt[0], dict)
        assert prompt[0]['content'][0]["type"] == "text"
        assert prompt[0]['content'][0]["text"] == "What's in these images?"

        assert isinstance(prompt[1], dict)
        assert prompt[1]['content'][0]["type"] == "image"
        assert "source" in prompt[1]['content'][0]
        assert prompt[1]['content'][0]["source"]["type"] == "base64"
        assert prompt[1]['content'][0]["source"]["media_type"] == "image/jpeg"
        assert isinstance(prompt[1]['content'][0]["source"]["data"], str)
        assert prompt[1]['content'][0]["source"]["data"].startswith("/")

    def test_generate_prompt_remote_image(self, mock_boto3_client, mock_key_handler):
        """Test multimodal prompt generation with a remote image."""
        model = MultimodalBedrockModel()

        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=TEST_IMAGE_URL, local=False)
        ]
        
        prompt = model.generate_prompt(multimodal_input)

        assert isinstance(prompt, list)
        assert len(prompt) == 2

        assert prompt[0]['content'][0]["type"] == "text"
        assert prompt[0]['content'][0]["text"] == "Describe this image:"

        assert isinstance(prompt[1], dict)
        assert prompt[1]['content'][0]["type"] == "image"
        assert "source" in prompt[1]['content'][0]
        assert prompt[1]['content'][0]["source"]["type"] == "base64"
        assert prompt[1]['content'][0]["source"]["media_type"] == "image/jpeg"
        assert isinstance(prompt[1]['content'][0]["source"]["data"], str)
        assert prompt[1]['content'][0]["source"]["data"].startswith("/")


    def test_generate(self, mock_boto3_client, mock_key_handler):
        """Test multimodal generation with image and text."""
        model = MultimodalBedrockModel()

        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=TEST_LOCAL_IMAGE, local=True)
        ]

        response = model.generate(multimodal_input)

        assert response == TEST_RESPONSE

        mock_instance = mock_boto3_client.return_value
        mock_instance.invoke_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_generate(self, mock_boto3_client, mock_key_handler):
        """Test async multimodal generation."""
        model = MultimodalBedrockModel()

        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=TEST_IMAGE_URL, local=False)
        ]

        response = await model.a_generate(multimodal_input)

        assert response == TEST_RESPONSE

        mock_instance = mock_boto3_client.return_value
        mock_instance.invoke_model.assert_called_once()

    def test_invalid_input_type(self, mock_boto3_client, mock_key_handler):
        """Test handling of invalid input types."""
        model = MultimodalBedrockModel()

        multimodal_input = [
            "Describe this image:",
            {"url": TEST_IMAGE_URL}
        ]

        with pytest.raises(ValueError, match="Invalid input type"):
            model.generate_prompt(multimodal_input)
