"""Tests for Anthropic prompt-caching token extraction in the output extractor."""

import pytest
from unittest.mock import MagicMock

from deepeval.anthropic.extractors import (
    extract_messages_api_output_parameters,
    safe_extract_output_parameters,
)
from deepeval.model_integrations.types import InputParameters


def _make_usage(
    input_tokens=100,
    output_tokens=50,
    cache_creation_input_tokens=None,
    cache_read_input_tokens=None,
):
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation_input_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens
    return usage


def _make_message(usage, content_text="Hello"):
    content_block = MagicMock()
    content_block.text = content_text
    message = MagicMock()
    message.content = [content_block]
    message.usage = usage
    return message


_input_params = InputParameters(model="claude-sonnet-4-5")


class TestCacheCreationTokens:
    def test_cache_creation_tokens_extracted_when_present(self):
        usage = _make_usage(cache_creation_input_tokens=200)
        message = _make_message(usage)

        params = extract_messages_api_output_parameters(message, _input_params)

        assert params.cache_creation_input_tokens == 200
        assert params.cache_read_input_tokens is None

    def test_cache_read_tokens_extracted_when_present(self):
        usage = _make_usage(cache_read_input_tokens=150)
        message = _make_message(usage)

        params = extract_messages_api_output_parameters(message, _input_params)

        assert params.cache_creation_input_tokens is None
        assert params.cache_read_input_tokens == 150

    def test_both_cache_token_fields_extracted_together(self):
        usage = _make_usage(
            cache_creation_input_tokens=300, cache_read_input_tokens=75
        )
        message = _make_message(usage)

        params = extract_messages_api_output_parameters(message, _input_params)

        assert params.cache_creation_input_tokens == 300
        assert params.cache_read_input_tokens == 75

    def test_no_cache_tokens_when_not_set(self):
        usage = _make_usage()
        message = _make_message(usage)

        params = extract_messages_api_output_parameters(message, _input_params)

        assert params.cache_creation_input_tokens is None
        assert params.cache_read_input_tokens is None

    def test_regular_token_counts_unaffected(self):
        usage = _make_usage(
            input_tokens=123,
            output_tokens=45,
            cache_creation_input_tokens=60,
        )
        message = _make_message(usage)

        params = extract_messages_api_output_parameters(message, _input_params)

        assert params.prompt_tokens == 123
        assert params.completion_tokens == 45

    def test_safe_extract_returns_cache_tokens_when_present(self):
        usage = _make_usage(
            cache_creation_input_tokens=100, cache_read_input_tokens=50
        )
        message = _make_message(usage)

        params = safe_extract_output_parameters(message, _input_params)

        assert params.cache_creation_input_tokens == 100
        assert params.cache_read_input_tokens == 50

    def test_getattr_guard_when_usage_lacks_cache_fields(self):
        """Usage objects from older SDK versions may not have cache fields at all."""
        usage = MagicMock(spec=["input_tokens", "output_tokens"])
        usage.input_tokens = 10
        usage.output_tokens = 5
        message = _make_message(usage)

        params = extract_messages_api_output_parameters(message, _input_params)

        assert params.cache_creation_input_tokens is None
        assert params.cache_read_input_tokens is None
