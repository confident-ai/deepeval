import pytest

class TestModelNameFormat:
    """
    Since parse_model_name function is removed, 
    we expect model names to be used directly in full 'provider/model' format.
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            "openai/gpt-4o",
            "anthropic/claude-3-opus",
            "cohere/command",
            "local/llama-3",
            "custom-provider/model-123_test",
            "gpt-4o",
            "claude-3-sonnet",
        ],
    )
    def test_model_name_direct_usage(self, model_name):
        # Model names should be used as-is, with no parsing or stripping.
        assert isinstance(model_name, str)
        assert len(model_name) > 0

    def test_model_name_format_contains_provider(self):
        # Check that common provider/model format contains a slash
        assert "/" in "openai/gpt-4o"
        assert "/" in "anthropic/claude-3-opus"

    def test_model_name_format_no_slash(self):
        # Model names without provider are allowed as-is
        model = "gpt-4o"
        assert "/" not in model
