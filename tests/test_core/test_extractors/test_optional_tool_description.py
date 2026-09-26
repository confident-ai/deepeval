"""Regression tests for optional tool ``description`` in extractor kwargs.

Both the Anthropic Messages API and the OpenAI Chat Completions / Responses
APIs treat a tool's ``description`` field as optional. When a user passes a
tool without one, input-parameter extraction must still succeed: previously a
``KeyError`` was raised, and because the ``safe_extract_*`` wrappers catch
everything, the whole trace silently degraded to ``InputParameters(model="NA")``
with no input, messages, tools or tool descriptions.
"""

import pytest

from deepeval.anthropic.extractors import (
    extract_messages_api_input_parameters,
    safe_extract_input_parameters as safe_extract_anthropic_input_parameters,
)
from deepeval.openai.extractors import (
    extract_input_parameters_from_completion,
    extract_input_parameters_from_response,
    safe_extract_input_parameters as safe_extract_openai_input_parameters,
)


class TestAnthropicOptionalToolDescription:
    def test_tool_without_description_does_not_raise(self):
        kwargs = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        }

        params = extract_messages_api_input_parameters(kwargs)

        assert params.model == "claude-sonnet-4-5"
        assert params.tool_descriptions == {}

    def test_mixed_tools_keep_descriptions_that_exist(self):
        kwargs = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"name": "with_desc", "description": "does things"},
                {"name": "without_desc"},
            ],
        }

        params = extract_messages_api_input_parameters(kwargs)

        assert params.tool_descriptions == {"with_desc": "does things"}
        assert "without_desc" not in params.tool_descriptions

    def test_safe_wrapper_no_longer_degrades_to_na(self):
        kwargs = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [{"name": "get_weather"}],
        }

        params = safe_extract_anthropic_input_parameters(kwargs)

        assert params.model == "claude-sonnet-4-5"
        assert params.input is not None


class TestOpenAIOptionalToolDescription:
    def test_completion_tool_without_description_does_not_raise(self):
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        }

        params = extract_input_parameters_from_completion(kwargs)

        assert params.model == "gpt-4o"
        assert params.tool_descriptions == {}

    def test_response_tool_without_description_does_not_raise(self):
        kwargs = {
            "model": "gpt-4o",
            "input": "Weather in Paris?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        }

        params = extract_input_parameters_from_response(kwargs)

        assert params.model == "gpt-4o"
        assert params.tool_descriptions == {}

    @pytest.mark.parametrize("is_completion", [True, False])
    def test_safe_wrapper_no_longer_degrades_to_na(self, is_completion):
        if is_completion:
            kwargs = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f"}}],
            }
        else:
            kwargs = {
                "model": "gpt-4o",
                "input": "hi",
                "tools": [{"type": "function", "name": "f"}],
            }

        params = safe_extract_openai_input_parameters(is_completion, kwargs)

        assert params.model == "gpt-4o"
