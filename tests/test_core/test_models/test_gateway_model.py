"""Client lifecycle tests for `DeepEvalOpenAICompatibleModel`.

Regression tests for https://github.com/confident-ai/deepeval/issues/3120:
the gateway base builds a fresh ``OpenAI`` / ``AsyncOpenAI`` client for every
call, and used to leave every one of them open. Leaked async clients are later
collected by the GC, whose cleanup schedules ``aclose()`` against connections
owned by an already-closed event loop, spraying
``RuntimeError: Event loop is closed`` tracebacks into user output.

These tests use real openai-SDK clients (only the HTTP ``create`` methods are
stubbed) and assert that every client the model creates is closed again before
the call returns.
"""

import httpx
import pytest
from unittest.mock import Mock, AsyncMock, patch

from openai.resources.chat.completions import AsyncCompletions, Completions

from deepeval.models.llms.openrouter_model import OpenRouterModel


def _fake_completion():
    completion = Mock()
    completion.choices = [Mock(message=Mock(content="test response"))]
    completion.usage.prompt_tokens = 10
    completion.usage.completion_tokens = 20
    return completion


class _ClientRecordingModel(OpenRouterModel):
    """OpenRouterModel that records every client ``load_model`` hands out."""

    def __init__(self, *args, **kwargs):
        self.created_clients = []
        super().__init__(*args, **kwargs)

    def load_model(self, async_mode: bool = False):
        client = super().load_model(async_mode)
        self.created_clients.append(client)
        return client


def _assert_all_clients_closed(model: _ClientRecordingModel):
    assert model.created_clients, "expected at least one client to be created"
    for client in model.created_clients:
        assert client.is_closed(), (
            "gateway model left an openai client open after the call; "
            "clients must be closed in the event loop that created them "
            "(issue #3120)"
        )


@pytest.fixture
def model():
    recording_model = _ClientRecordingModel(
        model="openai/gpt-4o-mini", api_key="test-key"
    )
    # DeepEvalBaseLLM.__init__ stores one client as `self.model` for the
    # lifetime of the instance; that one is referenced, not leaked. Only
    # clients created per call are asserted on.
    recording_model.created_clients.clear()
    return recording_model


class TestGatewayModelClientLifecycle:
    """Every client a gateway call creates must be closed before it returns."""

    async def test_a_generate_closes_async_client(self, model):
        with patch.object(
            AsyncCompletions,
            "create",
            new_callable=AsyncMock,
            return_value=_fake_completion(),
        ):
            output, _ = await model.a_generate("test prompt")

        assert output == "test response"
        _assert_all_clients_closed(model)

    def test_generate_closes_async_client_across_loops(self, model):
        # Sync path: each call runs in its own short-lived asyncio.run()
        # loop, which is exactly where leaked clients caused the GC-time
        # "Event loop is closed" tracebacks. Two calls mimic back-to-back
        # metrics with async_mode=False.
        with patch.object(
            AsyncCompletions,
            "create",
            new_callable=AsyncMock,
            return_value=_fake_completion(),
        ):
            output, _ = model.generate("test prompt")
            model.generate("another prompt")

        assert output == "test response"
        assert len(model.created_clients) == 2
        _assert_all_clients_closed(model)

    async def test_a_generate_raw_response_closes_async_client(self, model):
        with patch.object(
            AsyncCompletions,
            "create",
            new_callable=AsyncMock,
            return_value=_fake_completion(),
        ):
            completion, _ = await model.a_generate_raw_response("test prompt")

        assert completion.choices[0].message.content == "test response"
        _assert_all_clients_closed(model)

    def test_generate_raw_response_closes_sync_client(self, model):
        with patch.object(
            Completions, "create", return_value=_fake_completion()
        ):
            completion, _ = model.generate_raw_response("test prompt")

        assert completion.choices[0].message.content == "test response"
        _assert_all_clients_closed(model)

    def test_generate_samples_closes_sync_client(self, model):
        with patch.object(
            Completions, "create", return_value=_fake_completion()
        ):
            samples, _ = model.generate_samples(
                "test prompt", n=1, temperature=0.5
            )

        assert samples == ["test response"]
        _assert_all_clients_closed(model)

    def test_user_supplied_http_client_survives_repeated_calls(self):
        # When the user passes their own http_client, every per-call openai
        # client wraps that same transport; closing it after the first call
        # would break every later call, so the model must leave it open.
        http_client = httpx.Client()
        try:
            model = _ClientRecordingModel(
                model="openai/gpt-4o-mini",
                api_key="test-key",
                http_client=http_client,
            )
            model.created_clients.clear()
            with patch.object(
                Completions, "create", return_value=_fake_completion()
            ):
                model.generate_raw_response("test prompt")
                model.generate_raw_response("another prompt")

            assert not http_client.is_closed
        finally:
            http_client.close()

    async def test_a_generate_closes_async_client_on_error(self, model):
        # Called directly (not through the retry-wrapped public method) so
        # the assertion is independent of the provider retry policy.
        with patch.object(
            AsyncCompletions,
            "create",
            new_callable=AsyncMock,
            side_effect=ValueError("boom"),
        ):
            with pytest.raises(ValueError):
                await model._a_generate("test prompt")

        _assert_all_clients_closed(model)
