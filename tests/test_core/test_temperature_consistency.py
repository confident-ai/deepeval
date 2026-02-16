"""Verify that temperature is passed to the API in every generate code path.

These tests mock the underlying OpenAI-compatible client so no real API
calls are made.  They assert that ``temperature=<value>`` appears in
the keyword arguments of every ``chat.completions.create`` invocation,
regardless of whether a schema is supplied.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import BaseModel


# ── helpers ──────────────────────────────────────────────────────────


class _DummySchema(BaseModel):
    answer: str


def _fake_usage():
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    return usage


def _fake_completion(content='{"answer": "hello"}'):
    choice = MagicMock()
    choice.message.content = content
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = _fake_usage()
    return completion


# ── DeepSeek ─────────────────────────────────────────────────────────

_DS_INIT_KWARGS = dict(
    model="deepseek-chat",
    api_key="fake-key",
)


class TestDeepSeekTemperature:
    """DeepSeekModel must pass temperature in both schema and non-schema
    code paths for generate() and a_generate()."""

    def _make_model(self, temperature=0.3):
        from deepeval.models.llms.deepseek_model import DeepSeekModel

        return DeepSeekModel(temperature=temperature, **_DS_INIT_KWARGS)

    # -- sync ----------------------------------------------------------

    def test_generate_no_schema_passes_temperature(self):
        model = self._make_model()
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_completion(
            "plain text"
        )
        with patch.object(model, "load_model", return_value=client):
            model.generate(prompt="hi")

        kwargs = client.chat.completions.create.call_args
        assert (
            "temperature" in kwargs.kwargs
        ), "temperature missing from non-schema generate() call"
        assert kwargs.kwargs["temperature"] == pytest.approx(0.3)

    def test_generate_with_schema_passes_temperature(self):
        model = self._make_model()
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_completion()
        with patch.object(model, "load_model", return_value=client):
            model.generate(prompt="hi", schema=_DummySchema)

        kwargs = client.chat.completions.create.call_args
        assert "temperature" in kwargs.kwargs
        assert kwargs.kwargs["temperature"] == pytest.approx(0.3)

    # -- async ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_a_generate_no_schema_passes_temperature(self):
        model = self._make_model()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_completion("plain text")
        )
        with patch.object(model, "load_model", return_value=client):
            await model.a_generate(prompt="hi")

        kwargs = client.chat.completions.create.call_args
        assert "temperature" in kwargs.kwargs
        assert kwargs.kwargs["temperature"] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_a_generate_with_schema_passes_temperature(self):
        model = self._make_model()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_completion()
        )
        with patch.object(model, "load_model", return_value=client):
            await model.a_generate(prompt="hi", schema=_DummySchema)

        kwargs = client.chat.completions.create.call_args
        assert "temperature" in kwargs.kwargs
        assert kwargs.kwargs["temperature"] == pytest.approx(0.3)


# ── Kimi ─────────────────────────────────────────────────────────────

_KIMI_INIT_KWARGS = dict(
    model="moonshot-v1-8k",
    api_key="fake-key",
)


class TestKimiTemperature:
    """KimiModel must pass temperature in both schema and non-schema
    code paths for generate() and a_generate()."""

    def _make_model(self, temperature=0.5):
        from deepeval.models.llms.kimi_model import KimiModel

        return KimiModel(temperature=temperature, **_KIMI_INIT_KWARGS)

    # -- sync ----------------------------------------------------------

    def test_generate_no_schema_passes_temperature(self):
        model = self._make_model()
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_completion(
            "plain text"
        )
        with patch.object(model, "load_model", return_value=client):
            model.generate(prompt="hi")

        kwargs = client.chat.completions.create.call_args
        assert (
            "temperature" in kwargs.kwargs
        ), "temperature missing from non-schema generate() call"
        assert kwargs.kwargs["temperature"] == pytest.approx(0.5)

    def test_generate_with_schema_passes_temperature(self):
        model = self._make_model()
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_completion()
        with patch.object(model, "load_model", return_value=client):
            model.generate(prompt="hi", schema=_DummySchema)

        kwargs = client.chat.completions.create.call_args
        assert "temperature" in kwargs.kwargs
        assert kwargs.kwargs["temperature"] == pytest.approx(0.5)

    # -- async ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_a_generate_no_schema_passes_temperature(self):
        model = self._make_model()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_completion("plain text")
        )
        with patch.object(model, "load_model", return_value=client):
            await model.a_generate(prompt="hi")

        kwargs = client.chat.completions.create.call_args
        assert "temperature" in kwargs.kwargs
        assert kwargs.kwargs["temperature"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_a_generate_with_schema_passes_temperature(self):
        model = self._make_model()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_completion()
        )
        with patch.object(model, "load_model", return_value=client):
            await model.a_generate(prompt="hi", schema=_DummySchema)

        kwargs = client.chat.completions.create.call_args
        assert "temperature" in kwargs.kwargs
        assert kwargs.kwargs["temperature"] == pytest.approx(0.5)
