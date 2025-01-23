"""Tests for Gemini model implementations
"""

import pytest
from unittest.mock import patch, MagicMock
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image

from deepeval.models import GeminiModel, MultimodalGeminiModel
from deepeval.test_case import MLLMImage
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER

# Mock credentials for testing
TEST_PROJECT = "test-project"
TEST_LOCATION = "us-central1"
TEST_RESPONSE = "This is a test response"

@pytest.fixture
def mock_vertex_init():
    with patch('vertexai.init') as mock:
        yield mock

@pytest.fixture
def mock_part():
    with patch('deepeval.models.gemini_model.Part') as mock:
        mock.return_value = MagicMock(name="MockPart")
        mock.from_image.return_value = MagicMock(name="MockImagePart")
        mock.from_uri.return_value = MagicMock(name="MockUriPart")
        yield mock

@pytest.fixture
def mock_image():
    with patch('deepeval.models.gemini_model.Image') as mock:
        mock.load_from_file.return_value = MagicMock(name="MockFileImage")
        mock.from_url.return_value = MagicMock(name="MockUrlImage")
        yield mock

@pytest.fixture
def mock_generative_model():
    with patch('deepeval.models.gemini_model.GenerativeModel') as mock:
        instance = mock.return_value
        instance.generate_content.return_value = MagicMock(text=TEST_RESPONSE)
        
        # Create an async mock for generate_content_async
        async_response = MagicMock(text=TEST_RESPONSE)
        async def async_magic():
            return async_response
        instance.generate_content_async.return_value = async_magic()
        
        yield mock

@pytest.fixture
def mock_key_handler():
    with patch('deepeval.key_handler.KEY_FILE_HANDLER.fetch_data') as mock:
        mock.side_effect = lambda x: {
            KeyValues.GOOGLE_CLOUD_PROJECT: TEST_PROJECT,
            KeyValues.GOOGLE_CLOUD_LOCATION: TEST_LOCATION
        }.get(x)
        yield mock

class TestGeminiModel:
    """Test suite for text-only Gemini model"""

    def test_initialization(self, mock_vertex_init, mock_generative_model, mock_key_handler):
        """Test model initialization with default parameters"""
        model = GeminiModel()
        
        # Verify model initialization
        assert model.model_name == "gemini-1.5-pro"
        assert model.project_id == TEST_PROJECT
        assert model.location == TEST_LOCATION
        
        # Verify Vertex AI initialization
        mock_vertex_init.assert_called_once_with(
            project=TEST_PROJECT,
            location=TEST_LOCATION
        )
        
        # Verify GenerativeModel initialization
        mock_generative_model.assert_called_once()

    def test_initialization_with_custom_params(self, mock_vertex_init, mock_generative_model):
        """Test model initialization with custom parameters"""
        model = GeminiModel(
            model_name="gemini-1.5-flash",
            project_id="custom-project",
            location="europe-west4"
        )
        
        assert model.model_name == "gemini-1.5-flash"
        assert model.project_id == "custom-project"
        assert model.location == "europe-west4"

    def test_invalid_model_name(self):
        """Test initialization with invalid model name"""
        with pytest.raises(ValueError, match="Invalid model"):
            GeminiModel(model_name="invalid-model")

    def test_generate(self, mock_vertex_init, mock_generative_model, mock_key_handler):
        """Test text generation"""
        model = GeminiModel()
        test_prompt = "Test prompt"
        response = model.generate(test_prompt)
        
        # Verify response
        assert response == TEST_RESPONSE
        
        # Verify mock was called correctly
        mock_instance = mock_generative_model.return_value
        mock_instance.generate_content.assert_called_once_with(test_prompt)

    @pytest.mark.asyncio
    async def test_a_generate(self, mock_vertex_init, mock_generative_model, mock_key_handler):
        """Test async text generation"""
        model = GeminiModel()
        test_prompt = "Test prompt"
        response = await model.a_generate(test_prompt)
        
        # Verify response
        assert response == TEST_RESPONSE
        
        # Verify mock was called correctly
        mock_instance = mock_generative_model.return_value
        mock_instance.generate_content_async.assert_called_once_with(test_prompt)

class TestMultimodalGeminiModel:
    """Test suite for multimodal Gemini model"""

    def test_initialization(self, mock_vertex_init, mock_generative_model, mock_key_handler):
        """Test model initialization with default parameters"""
        model = MultimodalGeminiModel()
        
        # Verify model initialization
        assert model.model_name == "gemini-1.5-pro"
        assert model.project_id == TEST_PROJECT
        assert model.location == TEST_LOCATION
        
        # Verify Vertex AI initialization
        mock_vertex_init.assert_called_once_with(
            project=TEST_PROJECT,
            location=TEST_LOCATION
        )
        
        # Verify GenerativeModel initialization
        mock_generative_model.assert_called_once()

    def test_initialization_with_custom_params(self, mock_vertex_init, mock_generative_model):
        """Test model initialization with custom parameters"""
        model = MultimodalGeminiModel(
            model_name="gemini-1.0-pro-vision",
            project_id="custom-project",
            location="europe-west4"
        )
        
        assert model.model_name == "gemini-1.0-pro-vision"
        assert model.project_id == "custom-project"
        assert model.location == "europe-west4"

    def test_invalid_model_name(self):
        """Test initialization with invalid model name"""
        with pytest.raises(ValueError, match="Invalid model"):
            MultimodalGeminiModel(model_name="invalid-model")

    def test_generate_prompt_local_image(self, mock_vertex_init, mock_generative_model, mock_key_handler, mock_image, mock_part):
        """Test prompt generation with local image"""
        model = MultimodalGeminiModel()
        
        # Create test input
        test_file = "test.jpg"
        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=test_file, local=True)
        ]
        
        # Generate prompt
        prompt = model.generate_prompt(multimodal_input)
        
        # Verify prompt structure and mock calls
        assert len(prompt) == 2
        assert prompt[0] == "Describe this image:"
        assert isinstance(prompt[1], MagicMock)
        
        # Verify the chain of calls for local image
        mock_image.load_from_file.assert_called_once_with(test_file)
        loaded_image = mock_image.load_from_file.return_value
        mock_part.from_image.assert_called_once_with(loaded_image)

    def test_generate_prompt_remote_image(self, mock_vertex_init, mock_generative_model, mock_key_handler, mock_image, mock_part):
        """Test prompt generation with remote image"""
        model = MultimodalGeminiModel()
        
        # Create test input
        test_url = "https://example.com/test.jpg"
        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=test_url, local=False)
        ]
        
        # Generate prompt
        prompt = model.generate_prompt(multimodal_input)
        
        # Verify prompt structure and mock calls
        assert len(prompt) == 2
        assert prompt[0] == "Describe this image:"
        assert isinstance(prompt[1], MagicMock)
        mock_part.from_uri.assert_called_once_with(uri=test_url, mime_type="image/jpeg")

    def test_generate(self, mock_vertex_init, mock_generative_model, mock_key_handler, mock_image, mock_part):
        """Test multimodal generation"""
        model = MultimodalGeminiModel()
        
        # Create test input
        test_url = "https://example.com/test.jpg"
        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=test_url, local=False)
        ]
        
        # Get the expected prompt that will be passed to generate_content
        prompt = model.generate_prompt(multimodal_input)
        response = model.generate(multimodal_input)
        
        # Verify response
        assert response == TEST_RESPONSE
        
        # Verify mock calls
        mock_instance = mock_generative_model.return_value
        mock_instance.generate_content.assert_called_once_with(prompt)

    @pytest.mark.asyncio
    async def test_a_generate(self, mock_vertex_init, mock_generative_model, mock_key_handler, mock_image, mock_part):
        """Test async multimodal generation"""
        model = MultimodalGeminiModel()
        
        # Create test input
        test_url = "https://example.com/test.jpg"
        multimodal_input = [
            "Describe this image:",
            MLLMImage(url=test_url, local=False)
        ]
        
        # Get the expected prompt that will be passed to generate_content_async
        prompt = model.generate_prompt(multimodal_input)
        response = await model.a_generate(multimodal_input)
        
        # Verify response
        assert response == TEST_RESPONSE
        
        # Verify mock calls
        mock_instance = mock_generative_model.return_value
        mock_instance.generate_content_async.assert_called_once_with(prompt)

    def test_invalid_input_type(self, mock_vertex_init, mock_generative_model, mock_key_handler):
        """Test handling of invalid input types"""
        model = MultimodalGeminiModel()
        
        # Create test input with invalid type
        multimodal_input = [
            "Describe this image:",
            {"url": "test.jpg"}  # Invalid type
        ]
        
        with pytest.raises(ValueError, match="Invalid input type"):
            model.generate_prompt(multimodal_input)
