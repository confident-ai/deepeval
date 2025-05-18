import pytest
from deepeval.models.utils import parse_model_name


class TestGetActualModelName:
    """Test suite for the parse_model_name function."""

    @pytest.mark.parametrize(
        "input_model_name,expected_output",
        [
            # Standard provider/model format
            ("openai/gpt-4o", "gpt-4o"),
            ("anthropic/claude-3-opus", "claude-3-opus"),
            ("cohere/command", "command"),
            # No provider prefix
            ("gpt-4o", "gpt-4o"),
            ("claude-3-sonnet", "claude-3-sonnet"),
            # Edge cases
            ("", ""),  # Empty string
            ("/", ""),  # Just a slash
            ("openai/", ""),  # Provider with no model
            (
                "//model",
                "/model",
            ),  # Multiple slashes at start - splits only on first slash
            (
                "provider/model/version",
                "model/version",
            ),  # Multiple slashes in name
            (
                "provider/model-name/with/slashes",
                "model-name/with/slashes",
            ),  # Multiple slashes
            # Mixed case and special characters
            ("OpenAI/GPT-4o", "GPT-4o"),
            ("custom-provider/model-123_test", "model-123_test"),
            # Numerical and versioned names
            ("openai/gpt-3.5-turbo", "gpt-3.5-turbo"),
            ("anthropic/claude-2.1", "claude-2.1"),
        ],
    )
    def test_parse_model_name(self, input_model_name, expected_output):
        """
        Test that parse_model_name correctly extracts the model name
        from various input formats.

        Args:
            input_model_name: The model name to process
            expected_output: The expected result after processing
        """
        assert parse_model_name(input_model_name) == expected_output

    def test_parse_model_name_type_preservation(self):
        """Test that the function preserves the string type and doesn't modify the input."""
        result = parse_model_name("provider/model")
        assert isinstance(result, str)

        result = parse_model_name("")
        assert isinstance(result, str)

    def test_parse_model_name_identity(self):
        """Test that the function returns equal values for inputs without providers."""
        test_cases = ["gpt-4", "claude-3", "command-r", "model-with-dashes"]
        for model_name in test_cases:
            assert parse_model_name(model_name) == model_name

    def test_parse_model_name_none_value(self):
        """Test that the function raises TypeError when None is passed."""
        with pytest.raises(TypeError):
            parse_model_name(None)
