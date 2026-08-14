"""Tests for AnthropicModel's max_tokens default and truncation reporting.

Kept separate from test_anthropic_model.py so this PR shares no modified
test file with #2947 (which appends its own section there).
"""

import asyncio

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from deepeval.errors import DeepEvalError
from deepeval.models.llms.anthropic_model import AnthropicModel

from tests.test_core.stubs import _RecordingClient


class _TruncationVerdict(BaseModel):
    reason: str


def _make_message(stop_reason, text):
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=SimpleNamespace(input_tokens=10, output_tokens=100),
    )


class _SyncMessagesClient(_RecordingClient):
    """Recording client whose messages.create returns a canned message."""

    response = _make_message("end_turn", '{"reason": "ok"}')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **create_kwargs):
        self.create_kwargs = create_kwargs
        return type(self).response


class _AsyncMessagesClient(_SyncMessagesClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        async def _a_create(**create_kwargs):
            self.create_kwargs = create_kwargs
            return type(self).response

        self.messages = SimpleNamespace(create=_a_create)


def _fake_module(sync_cls, async_cls=None):
    return SimpleNamespace(
        Anthropic=sync_cls, AsyncAnthropic=async_cls or sync_cls
    )


def _configure_settings(settings):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "test-key"
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = 1e-6
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = 1e-6


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_defaults_max_tokens_to_8192(
    mock_require_dep, settings
):
    """1024 starved thinking-by-default models (e.g. claude-opus-5), where
    max_tokens caps thinking + response text combined."""
    _configure_settings(settings)
    mock_require_dep.return_value = _fake_module(_RecordingClient)

    model = AnthropicModel(model="claude-opus-5")
    assert model._max_tokens == 8192


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_explicit_max_tokens_still_wins(
    mock_require_dep, settings
):
    _configure_settings(settings)
    mock_require_dep.return_value = _fake_module(_RecordingClient)

    model = AnthropicModel(model="claude-opus-5", max_tokens=512)
    assert model._max_tokens == 512

    model = AnthropicModel(
        model="claude-opus-5", generation_kwargs={"max_tokens": 2048}
    )
    assert model._max_tokens == 2048


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_generate_raises_clear_error_when_truncated_with_schema(
    mock_require_dep, settings
):
    """A response cut off at max_tokens used to surface downstream as an
    invalid-JSON error; it should name max_tokens instead."""
    _configure_settings(settings)

    class _TruncatedClient(_SyncMessagesClient):
        response = _make_message("max_tokens", '{"reason": "truncat')

    mock_require_dep.return_value = _fake_module(_TruncatedClient)

    model = AnthropicModel(model="claude-opus-5")
    with pytest.raises(DeepEvalError, match="max_tokens"):
        model.generate("judge this", schema=_TruncationVerdict)


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_a_generate_raises_clear_error_when_truncated_with_schema(
    mock_require_dep, settings
):
    _configure_settings(settings)

    class _TruncatedAsyncClient(_AsyncMessagesClient):
        response = _make_message("max_tokens", '{"reason": "truncat')

    mock_require_dep.return_value = _fake_module(
        _RecordingClient, _TruncatedAsyncClient
    )

    model = AnthropicModel(model="claude-opus-5")
    with pytest.raises(DeepEvalError, match="max_tokens"):
        asyncio.run(model.a_generate("judge this", schema=_TruncationVerdict))


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_generate_truncated_without_schema_returns_partial_text(
    mock_require_dep, settings
):
    """Plain-text callers may cap output deliberately; keep partial text."""
    _configure_settings(settings)

    class _TruncatedTextClient(_SyncMessagesClient):
        response = _make_message("max_tokens", "partial answer")

    mock_require_dep.return_value = _fake_module(_TruncatedTextClient)

    model = AnthropicModel(model="claude-opus-5")
    text, _ = model.generate("summarize")
    assert text == "partial answer"


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_generate_truncated_without_schema_and_no_text_raises(
    mock_require_dep, settings
):
    """All 8192 tokens spent on thinking: no text block at all."""
    _configure_settings(settings)

    class _EmptyTruncatedClient(_SyncMessagesClient):
        response = _make_message("max_tokens", None)

    mock_require_dep.return_value = _fake_module(_EmptyTruncatedClient)

    model = AnthropicModel(model="claude-opus-5")
    with pytest.raises(DeepEvalError, match="max_tokens"):
        model.generate("summarize")
