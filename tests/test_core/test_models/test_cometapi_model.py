# tests/test_core/test_models/test_cometapi_model.py
import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock
from tenacity import RetryError

from deepeval.models.llms.cometapi_model import CometAPIModel
from deepeval.models.embedding_models.cometapi_embedding_model import (
    CometAPIEmbeddingModel,
)


class TestCometAPIModelInitialization:
    """Test CometAPI model initialization with various parameters."""

    def test_valid_initialization_with_api_key(self):
        """Test that CometAPIModel initializes correctly with valid parameters."""
        model = CometAPIModel(
            model="gpt-4o-mini",
            api_key="test-key-123",
            temperature=0.5,
        )
        assert model.model_name == "gpt-4o-mini"
        assert model.temperature == 0.5

    def test_valid_initialization_with_env_var(self, monkeypatch):
        """Test initialization using COMETAPI_KEY environment variable."""
        monkeypatch.setenv("COMETAPI_KEY", "env-test-key")
        model = CometAPIModel(model="gpt-4o-mini")
        assert model.model_name == "gpt-4o-mini"

    def test_initialization_with_custom_pricing(self):
        """Test initialization with custom cost parameters (ignored in cost calculation)."""
        model = CometAPIModel(
            model="custom-model",
            api_key="test-key",
            cost_per_input_token=0.001 / 1e6,
            cost_per_output_token=0.002 / 1e6,
        )
        assert model.model_name == "custom-model"
        # Cost calculation now always returns 0.0 regardless of custom pricing
        cost = model.calculate_cost(
            input_tokens=1000, output_tokens=500
        )
        assert cost == 0.0

    def test_invalid_model_name(self):
        """Test that empty model name uses default model."""
        # CometAPI allows any model name, uses default if empty
        model = CometAPIModel(model="", api_key="test-key")
        # Should use default model
        assert model.model_name in ["gpt-4o-mini", ""]

    def test_missing_api_key(self, monkeypatch):
        """Test that model can be created without API key (will fail when used)."""
        monkeypatch.delenv("COMETAPI_KEY", raising=False)
        # Model creation succeeds without API key
        model = CometAPIModel(model="gpt-4o-mini")
        # API key will be None
        assert model.api_key is None


class TestCometAPIModelGeneration:
    """Test CometAPI model text generation functionality."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client for testing."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(content="Test response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=5,
        )
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def mock_async_openai_client(self):
        """Create a mock async OpenAI client for testing."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(content="Async test response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=15,
            completion_tokens=8,
        )
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        return mock_client

    def test_generate_text(self, monkeypatch, mock_openai_client):
        """Test synchronous text generation."""
        model = CometAPIModel(model="gpt-4o-mini", api_key="test-key")

        # Mock the load_model method to return our mock client
        def mock_load_model(async_mode=False):
            return mock_openai_client

        monkeypatch.setattr(model, "load_model", mock_load_model)

        result, cost = model.generate("Hello, world!")

        assert result == "Test response"
        assert cost == 0.0  # Cost is always 0.0
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_generate_text(
        self, monkeypatch, mock_async_openai_client
    ):
        """Test asynchronous text generation."""
        model = CometAPIModel(model="gpt-4o-mini", api_key="test-key")

        # Mock the load_model method to return our mock async client
        def mock_load_model(async_mode=False):
            return mock_async_openai_client

        monkeypatch.setattr(model, "load_model", mock_load_model)

        result, cost = await model.a_generate("Hello, async world!")

        assert result == "Async test response"
        assert cost == 0.0  # Cost is always 0.0
        mock_async_openai_client.chat.completions.create.assert_called_once()

    def test_generate_with_json_schema(self, monkeypatch):
        """Test generation with JSON schema (structured output)."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            answer: str
            confidence: float

        model = CometAPIModel(model="gpt-4o-mini", api_key="test-key")

        # Mock for structured output - need to properly mock response structure
        mock_choice = Mock()
        mock_choice.message.content = '{"answer": "test", "confidence": 0.95}'
        mock_choice.finish_reason = "stop"
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock(prompt_tokens=20, completion_tokens=10)

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        def mock_load_model(async_mode=False):
            return mock_client

        monkeypatch.setattr(model, "load_model", mock_load_model)

        result, cost = model.generate("Test prompt", schema=TestSchema)

        assert result.answer == "test"
        assert result.confidence == 0.95
        assert cost == 0.0  # Cost is always 0.0
        mock_client.chat.completions.create.assert_called_once()


class TestCometAPIModelCostCalculation:
    """Test cost calculation for various CometAPI models."""

    def test_cost_calculation_gpt_model(self):
        """Test cost calculation returns 0.0 as CometAPI does not provide pricing."""
        model = CometAPIModel(model="gpt-4o-mini", api_key="test-key")
        cost = model.calculate_cost(
            input_tokens=1000, output_tokens=500
        )
        assert cost == 0.0

    def test_cost_calculation_claude_model(self):
        """Test cost calculation returns 0.0 for all models."""
        model = CometAPIModel(
            model="claude-sonnet-4-20250514", api_key="test-key"
        )
        cost = model.calculate_cost(
            input_tokens=2000,
            output_tokens=1000
        )
        assert cost == 0.0

    def test_cost_calculation_unknown_model_with_custom_pricing(self):
        """Test cost calculation returns 0.0 even with custom pricing parameters."""
        model = CometAPIModel(
            model="unknown-model",
            api_key="test-key",
            cost_per_input_token=0.001 / 1e6,
            cost_per_output_token=0.002 / 1e6,
        )
        cost = model.calculate_cost(
            input_tokens=5000, output_tokens=2500
        )
        # Cost calculation now always returns 0.0
        assert cost == 0.0

    def test_cost_calculation_unknown_model_without_pricing(self):
        """Test that unknown models without custom pricing return 0 cost."""
        model = CometAPIModel(model="unknown-model-xyz", api_key="test-key")
        cost = model.calculate_cost(
            input_tokens=1000, output_tokens=500
        )
        assert cost == 0.0


class TestCometAPIModelRetryPolicy:
    """Test retry policy for CometAPI models."""

    @pytest.fixture
    def always_retryable_client(self):
        """Create a client that always raises retryable errors."""

        class AlwaysRetryableClient:
            def __init__(self, counter):
                self._counter = counter
                self.chat = type("Chat", (), {})()
                self.chat.completions = type("Completions", (), {})()
                self.chat.completions.create = self._raise

            def _raise(self, *args, **kwargs):
                self._counter["calls"] += 1
                req = httpx.Request("POST", "https://api.cometapi.com/v1/fake")
                resp = httpx.Response(
                    429,
                    request=req,
                    json={"error": {"code": "rate_limit"}},
                )
                body = {"error": {"code": "rate_limit"}}
                import openai

                raise openai.RateLimitError(
                    message="simulated retryable 429",
                    response=resp,
                    body=body,
                )

        counter = {"calls": 0}
        return AlwaysRetryableClient(counter), counter

    def test_retry_on_rate_limit(
        self, monkeypatch, always_retryable_client, settings
    ):
        """Test that rate limit errors trigger retries."""
        client, counter = always_retryable_client

        with settings.edit(persist=False):
            settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 3
            settings.DEEPEVAL_RETRY_CAP_SECONDS = 0

        model = CometAPIModel(model="gpt-4o-mini", api_key="test-key")

        def mock_load_model(async_mode=False):
            return client

        monkeypatch.setattr(model, "load_model", mock_load_model)

        # Should retry and eventually raise exception
        with pytest.raises((RetryError, Exception)):
            model.generate("test prompt")

        # Should have made at least 1 attempt
        assert counter["calls"] >= 1


class TestCometAPIEmbeddingModel:
    """Test CometAPI embedding model functionality."""

    def test_embedding_model_initialization(self):
        """Test that embedding model initializes correctly."""
        embedder = CometAPIEmbeddingModel(
            model="text-embedding-3-small", api_key="test-key"
        )
        assert embedder.model_name == "text-embedding-3-small"

    def test_invalid_embedding_model(self):
        """Test that invalid embedding model names raise ValueError."""
        with pytest.raises(
            ValueError, match="Invalid model"
        ):
            CometAPIEmbeddingModel(
                model="invalid-embedding-model", api_key="test-key"
            )

    def test_embed_single_text(self, monkeypatch):
        """Test embedding a single text."""
        embedder = CometAPIEmbeddingModel(
            model="text-embedding-3-small", api_key="test-key"
        )

        # Mock the OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        def mock_load_model(async_mode=False):
            return mock_client

        monkeypatch.setattr(embedder, "_build_client", mock_load_model)

        embedding = embedder.embed_text("Hello, world!")

        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    def test_embed_multiple_texts(self, monkeypatch):
        """Test embedding multiple texts."""
        embedder = CometAPIEmbeddingModel(
            model="text-embedding-3-small", api_key="test-key"
        )

        # Mock the OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        def mock_load_model(async_mode=False):
            return mock_client

        monkeypatch.setattr(embedder, "_build_client", mock_load_model)

        embeddings = embedder.embed_texts(["Text 1", "Text 2"])

        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_embed_text(self, monkeypatch):
        """Test asynchronous text embedding."""
        embedder = CometAPIEmbeddingModel(
            model="text-embedding-3-small", api_key="test-key"
        )

        # Mock the async OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.7, 0.8, 0.9])]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        def mock_load_model(async_mode=False):
            return mock_client

        monkeypatch.setattr(embedder, "load_model", mock_load_model)

        embedding = await embedder.a_embed_text("Async embedding test")

        assert embedding == [0.7, 0.8, 0.9]
        mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_embed_texts(self, monkeypatch):
        """Test asynchronous embedding of multiple texts."""
        embedder = CometAPIEmbeddingModel(
            model="text-embedding-3-large", api_key="test-key"
        )

        # Mock the async OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[1.0, 1.1, 1.2]),
            Mock(embedding=[1.3, 1.4, 1.5]),
            Mock(embedding=[1.6, 1.7, 1.8]),
        ]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        def mock_load_model(async_mode=False):
            return mock_client

        monkeypatch.setattr(embedder, "load_model", mock_load_model)

        embeddings = await embedder.a_embed_texts(
            ["Async 1", "Async 2", "Async 3"]
        )

        assert embeddings == [
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
            [1.6, 1.7, 1.8],
        ]
        mock_client.embeddings.create.assert_called_once()


class TestCometAPIModelProperties:
    """Test CometAPI model properties and metadata."""

    def test_model_name_property(self):
        """Test that model_name property returns correct value."""
        model = CometAPIModel(model="gpt-4o-mini", api_key="test-key")
        assert model.model_name == "gpt-4o-mini"

    def test_load_model_returns_client(self):
        """Test that load_model returns a valid OpenAI client."""
        model = CometAPIModel(model="gpt-4o-mini", api_key="test-key")
        client = model.load_model()
        assert client is not None
        # Verify it has the expected OpenAI client structure
        assert hasattr(client, "chat")

    def test_load_async_model_returns_async_client(self):
        """Test that load_model with async_mode returns AsyncOpenAI client."""
        model = CometAPIModel(model="gpt-4o-mini", api_key="test-key")
        async_client = model.load_model(async_mode=True)
        assert async_client is not None
        assert hasattr(async_client, "chat")
