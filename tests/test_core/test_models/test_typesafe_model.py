"""Tests for TypeSafeModel (Jev) initialization, generation, and schema support."""

import asyncio
from typing import List
from unittest.mock import Mock, patch, MagicMock
import pytest
from pydantic import BaseModel, SecretStr

from deepeval.config.settings import get_settings, reset_settings
from deepeval.errors import DeepEvalError
from deepeval.models.llms.typesafe_model import TypeSafeModel
from deepeval.models import DeepEvalBaseLLM


class SampleSchema(BaseModel):
    score: float
    reason: str


class StepsSchema(BaseModel):
    steps: List[str]


class MockAnswer:
    def __init__(self, choice=None, score=None, noul=None):
        self.choice = choice
        self.score = score
        self.noul = noul


class MockSystemOneResponse:
    def __init__(self, answers):
        self.answers = answers


class TestTypeSafeModelInit:
    def test_init_default_model(self):
        """TypeSafeModel should default to 'jev' when model is not provided."""
        model = TypeSafeModel(api_key="test-key")
        assert model.name == "jev"
        assert model.base_url == "https://api.typesafe.ai"
        assert model.generator_model is None

    def test_init_with_custom_model_and_url(self):
        """TypeSafeModel should accept custom model names and base_url."""
        mock_generator = Mock(spec=DeepEvalBaseLLM)
        model = TypeSafeModel(
            model="typesafe/jev",
            api_key="test-key",
            base_url="https://custom.gateway.ai",
            generator_model=mock_generator,
            generation_kwargs={"timeout": 10},
        )
        assert model.name == "typesafe/jev"
        assert model.base_url == "https://custom.gateway.ai"
        assert model.generator_model is mock_generator
        assert model.generation_kwargs == {"timeout": 10}

    def test_capabilities(self):
        """TypeSafeModel reports correct capabilities."""
        model = TypeSafeModel(api_key="test-key")
        assert model.supports_structured_outputs() is True
        assert model.supports_json_mode() is True
        assert model.supports_log_probs() is False
        assert model.supports_temperature() is False


class TestTypeSafeModelSecretManagement:
    def test_explicit_key_over_settings(self, monkeypatch):
        """Explicit api_key in ctor must take precedence over environment settings."""
        monkeypatch.setenv("TYPESAFE_API_KEY", "env-api-key")
        reset_settings(reload_dotenv=False)
        settings = get_settings()

        assert isinstance(settings.TYPESAFE_API_KEY, SecretStr)

        model = TypeSafeModel(api_key="explicit-key")
        assert model.api_key.get_secret_value() == "explicit-key"


class TestTypeSafeModelGeneration:
    def test_generate_text(self):
        """generate() should call client.system_one and return string output and cost."""
        mock_client = Mock()
        mock_response = MockSystemOneResponse({
            "evaluation": MockAnswer(score="Pass")
        })
        mock_client.system_one.return_value = mock_response

        model = TypeSafeModel(api_key="test-key")
        with patch.object(model, "load_model", return_value=mock_client):
            output, cost = model.generate("Evaluate this task")
            assert "Pass" in output
            assert isinstance(cost, (int, float))
            mock_client.system_one.assert_called_once()

    def test_generate_with_decision_schema(self):
        """generate() with decision schema should invoke Jev system_one."""
        mock_client = Mock()
        mock_response = MockSystemOneResponse({
            "score": MockAnswer(score="5"),
            "reason": MockAnswer(choice="High quality response"),
        })
        mock_client.system_one.return_value = mock_response

        model = TypeSafeModel(api_key="test-key")
        with patch.object(model, "load_model", return_value=mock_client):
            output, cost = model.generate("Rate this answer", schema=SampleSchema)
            assert isinstance(output, SampleSchema)
            assert output.score == 5.0
            assert "TypeSafe Jev model" in output.reason

    def test_generate_steps_schema_fallback(self):
        """generate() for synthesis StepsSchema without generator_model returns deterministic steps."""
        model = TypeSafeModel(api_key="test-key")
        output, cost = model.generate("Generate evaluation steps", schema=StepsSchema)
        assert isinstance(output, StepsSchema)
        assert len(output.steps) >= 2
        assert any("criteria" in s.lower() for s in output.steps)

    def test_generate_steps_schema_delegates_to_generator(self):
        """generate() for synthesis StepsSchema with generator_model delegates to generator."""
        mock_generator = Mock(spec=DeepEvalBaseLLM)
        expected_steps = StepsSchema(steps=["Custom Step 1", "Custom Step 2"])
        mock_generator.generate.return_value = (expected_steps, 0.001)

        model = TypeSafeModel(api_key="test-key", generator_model=mock_generator)
        output, cost = model.generate("Generate evaluation steps", schema=StepsSchema)
        
        mock_generator.generate.assert_called_once_with("Generate evaluation steps", schema=StepsSchema)
        assert output == expected_steps

    def test_generate_freeform_text_delegates_to_generator(self):
        """generate() freeform text with generator_model delegates to generator."""
        mock_generator = Mock(spec=DeepEvalBaseLLM)
        mock_generator.generate.return_value = ("Generated prose response", 0.002)

        model = TypeSafeModel(api_key="test-key", generator_model=mock_generator)
        output = model.generate("Write a paragraph")
        
        mock_generator.generate.assert_called_once_with("Write a paragraph")
        assert output == ("Generated prose response", 0.002)

    @pytest.mark.asyncio
    async def test_a_generate(self):
        """a_generate() should asynchronously return result and cost."""
        mock_client = Mock()
        mock_response = MockSystemOneResponse({
            "evaluation": MockAnswer(score="Pass")
        })
        mock_client.system_one.return_value = mock_response

        model = TypeSafeModel(api_key="test-key")
        with patch.object(model, "load_model", return_value=mock_client):
            output, cost = await model.a_generate("Async eval test")
            assert "Pass" in output
            assert isinstance(cost, (int, float))


class CitationSchema(BaseModel):
    citation: str
    reason: str


class TestTypeSafeModelMissingSDK:
    def test_missing_sdk_raises_error(self):
        """Should raise DeepEvalError with instructions if typesafe_sdk is missing."""
        model = TypeSafeModel(api_key="test-key")
        with patch.dict("sys.modules", {"typesafe_sdk": None}):
            with pytest.raises(DeepEvalError, match="pip install typesafe-sdk"):
                model.load_model()


class TestTypeSafeModelAliasesAndCitations:
    def test_alias_is_identical(self):
        """JevModel must be an identical alias of TypeSafeModel."""
        from deepeval.models import JevModel
        assert JevModel is TypeSafeModel

    def test_generate_citation_verification(self):
        """generate() with citation schema should map to 3-way Choice verification."""
        mock_client = Mock()
        mock_response = MockSystemOneResponse({
            "citation": MockAnswer(choice="supports"),
            "reason": MockAnswer(choice="Context fully verifies the statement"),
        })
        mock_client.system_one.return_value = mock_response

        model = TypeSafeModel(api_key="test-key")
        with patch.object(model, "load_model", return_value=mock_client):
            output, cost = model.generate("Verify citation relation", schema=CitationSchema)
            assert isinstance(output, CitationSchema)
            assert output.citation == "supports"
