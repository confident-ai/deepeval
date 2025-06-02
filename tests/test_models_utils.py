import pytest
from deepeval.models.utils import parse_model_name


class TestGetActualModelName:
    """Test suite for the parse_model_name function."""

    @pytest.mark.parametrize(
        "input_model_name,expected_output",
        [
            # Standard provider/model format - prefix stripped
            ("openai/gpt-4o", "gpt-4o"),
            ("anthropic/claude-3-opus", "claude-3-opus"),
            ("cohere/command", "command"),
            ("OpenAI/GPT-4o", "GPT-4o"),  # Case insensitive provider

            # No provider prefix - returns as is
            ("gpt-4o", "gpt-4o"),
            ("claude-3-sonnet", "claude-3-sonnet"),

            # Local/custom providers - preserved full string
            ("local/llama-3", "local/llama-3"),
            ("custom-provider/model-123_test", "custom-provider/model-123_test"),
            ("mymodels/awesome-llm", "mymodels/awesome-llm"),

            # Edge cases
            ("", ""),  # Empty string
            ("/", ""),  # Just a slash, treated as no provider name and empty model
            ("openai/", ""),  # Known provider with no model
            ("//model", "/model"),  # Multiple slashes at start - splits only on first slash
            ("provider/model/version", "provider/model/version"),  # Unknown provider, no stripping
            ("provider/model-name/with/slashes", "provider/model-name/with/slashes"),  # Unknown provider, no stripping

            # Numerical and versioned names with known providers
            ("openai/gpt-3.5-turbo", "gpt-3.5-turbo"),
            ("anthropic/claude-2.1", "claude-2.1"),
        ],
    )
    def test_parse_model_name(self, input_model_name, expected_output):
        """
        Test that parse_model_name correctly extracts the model name
        or preserves it for unknown prefixes.
        """
        assert parse_model_name(input_model_name) == expected_output

    def test_parse_model_name_type_preservation(self):
        """Test that the function preserves the string type and doesn't modify the input."""
        result = parse_model_name("provider/model")
        assert isinstance(result, str)

        result = parse_model_name("")
        assert isinstance(result, str)

    def test_parse_model_name_identity(self):
        """Test that the function returns equal values for inputs without known providers."""
        test_cases = [
            "gpt-4",
            "claude-3",
            "command-r",
            "model-with-dashes",
            "local/llama-3",
            "custom/model-xyz",
        ]
        for model_name in test_cases:
            assert parse_model_name(model_name) == model_name

    def test_parse_model_name_none_value(self):
        """Test that the function returns None when None is passed."""
        assert parse_model_name(None) is None
