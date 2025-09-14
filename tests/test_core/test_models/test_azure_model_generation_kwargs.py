"""Tests for AzureOpenAIModel generation_kwargs parameter"""

from unittest.mock import Mock, patch
from pydantic import BaseModel
import pytest
from deepeval.models.llms.azure_model import AzureOpenAIModel


class SampleSchema(BaseModel):
    """Sample schema for structured output testing"""

    field1: str
    field2: int


class TestAzureOpenAIModelGenerationKwargs:
    """Test suite for AzureOpenAIModel generation_kwargs functionality"""

    def test_init_without_generation_kwargs(self):
        """Test that AzureOpenAIModel initializes correctly without generation_kwargs"""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "test-endpoint",
                "AZURE_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_MODEL_NAME": "gpt-4",
                "OPENAI_API_VERSION": "2024-02-15-preview",
            },
        ):
            model = AzureOpenAIModel()
            assert model.generation_kwargs == {}
            assert model.kwargs == {}

    def test_init_with_generation_kwargs(self):
        """Test that AzureOpenAIModel initializes correctly with generation_kwargs"""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "test-endpoint",
                "AZURE_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_MODEL_NAME": "gpt-4",
                "OPENAI_API_VERSION": "2024-02-15-preview",
            },
        ):
            generation_kwargs = {
                "max_tokens": 1000,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
            }
            model = AzureOpenAIModel(generation_kwargs=generation_kwargs)
            assert model.generation_kwargs == generation_kwargs
            assert model.kwargs == {}

    def test_init_with_both_client_and_generation_kwargs(self):
        """Test that client kwargs and generation_kwargs are kept separate"""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "test-endpoint",
                "AZURE_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_MODEL_NAME": "gpt-4",
                "OPENAI_API_VERSION": "2024-02-15-preview",
            },
        ):
            generation_kwargs = {"max_tokens": 500}
            model = AzureOpenAIModel(
                generation_kwargs=generation_kwargs,
                timeout=30,  # client kwarg
                max_retries=3,  # client kwarg
            )
            assert model.generation_kwargs == generation_kwargs
            assert model.kwargs == {"timeout": 30, "max_retries": 3}

    @patch("deepeval.models.llms.azure_model.AzureOpenAIModel.load_model")
    def test_generate_with_generation_kwargs(self, mock_load_model):
        """Test that generation_kwargs are passed to generate method"""
        # Setup mock
        mock_client = Mock()
        mock_load_model.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="test response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_completion

        # Create model with explicit deployment_name to avoid KEY_FILE_HANDLER issues
        model = AzureOpenAIModel(
            deployment_name="test-deployment",
            model_name="gpt-4",
            azure_openai_api_key="test-key",
            azure_endpoint="test-endpoint",
            openai_api_version="2024-02-15-preview",
            generation_kwargs={"max_tokens": 1000, "top_p": 0.9},
        )

        # Call generate
        output, cost = model.generate("test prompt")

        # Verify the completion was called with generation_kwargs
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-deployment",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0,
            max_tokens=1000,
            top_p=0.9,
        )
        assert output == "test response"

    @patch("deepeval.models.llms.azure_model.AzureOpenAIModel.load_model")
    def test_generate_without_generation_kwargs(self, mock_load_model):
        """Test that generate works without generation_kwargs"""
        # Setup mock
        mock_client = Mock()
        mock_load_model.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="test response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_completion

        # Create model with explicit deployment_name to avoid KEY_FILE_HANDLER issues
        model = AzureOpenAIModel(
            deployment_name="test-deployment",
            model_name="gpt-4",
            azure_openai_api_key="test-key",
            azure_endpoint="test-endpoint",
            openai_api_version="2024-02-15-preview",
        )

        # Call generate without generation_kwargs
        output, cost = model.generate("test prompt")

        # Verify the completion was called without extra kwargs
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-deployment",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0,
        )
        assert output == "test response"

    @patch("deepeval.models.llms.azure_model.AzureOpenAI")
    def test_load_model_passes_kwargs_to_client(self, mock_azure_openai):
        """Test that client kwargs are passed, and SDK retries are disabled"""
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client

        model = AzureOpenAIModel(
            deployment_name="test-deployment",
            model_name="gpt-4",
            azure_openai_api_key="test-key",
            azure_endpoint="test-endpoint",
            openai_api_version="2024-02-15-preview",
            timeout=30,
            max_retries=5,  # user-provided, but we should override it to 0
        )

        mock_azure_openai.reset_mock()

        _ = model.load_model(async_mode=False)

        mock_azure_openai.assert_called_once()
        call_kwargs = mock_azure_openai.call_args[1]

        assert call_kwargs["timeout"] == 30
        # deepeval disables SDK retries to avoid double retries (Tenacity handles them)
        assert call_kwargs["max_retries"] == 0

    def test_backwards_compatibility(self):
        """Test that existing code without generation_kwargs still works"""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "test-endpoint",
                "AZURE_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_MODEL_NAME": "gpt-4",
                "OPENAI_API_VERSION": "2024-02-15-preview",
            },
        ):
            # This should work exactly as before
            model = AzureOpenAIModel(
                temperature=0.5, timeout=30  # client kwarg
            )
            assert model.temperature == 0.5
            assert model.kwargs == {"timeout": 30}
            assert model.generation_kwargs == {}

    def test_empty_generation_kwargs(self):
        """Test that empty generation_kwargs dict works correctly"""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "test-endpoint",
                "AZURE_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_MODEL_NAME": "gpt-4",
                "OPENAI_API_VERSION": "2024-02-15-preview",
            },
        ):
            model = AzureOpenAIModel(generation_kwargs={})
            assert model.generation_kwargs == {}

    def test_none_generation_kwargs(self):
        """Test that None generation_kwargs is handled correctly"""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "test-endpoint",
                "AZURE_DEPLOYMENT_NAME": "test-deployment",
                "AZURE_MODEL_NAME": "gpt-4",
                "OPENAI_API_VERSION": "2024-02-15-preview",
            },
        ):
            model = AzureOpenAIModel(generation_kwargs=None)
            assert model.generation_kwargs == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
