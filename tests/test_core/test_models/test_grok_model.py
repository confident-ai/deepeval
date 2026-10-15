"""Tests for GrokModel settings + secret handling (GROK_* + TEMPERATURE)."""

import asyncio
import sys
import types
from types import SimpleNamespace

import deepeval.models.llms.grok_model as grok_mod

from pydantic import SecretStr

from deepeval.models.llms.grok_model import GrokModel
from tests.test_core.stubs import _RecordingClient


def _stub_load_model(monkeypatch):
    """Avoid importing xai_sdk in tests by stubbing load_model."""

    def fake_load_model(self, async_mode: bool = False):
        return _RecordingClient()

    monkeypatch.setattr(
        grok_mod.GrokModel,
        "load_model",
        fake_load_model,
        raising=True,
    )


def _make_fake_xai_sdk():
    """
    Build a minimal fake xai_sdk module tree so that generate() and
    a_generate() can be exercised without the real SDK installed.

    The fake ``user`` function records the positional arguments it receives,
    which lets tests assert that we pass a plain str (not a list-of-dicts).
    """
    captured = {}

    def fake_user(*args):
        captured["args"] = args
        return SimpleNamespace(_args=args)

    def fake_text(s):
        return SimpleNamespace(type="text", text=s)

    def fake_image(url):
        return SimpleNamespace(type="image", url=url)

    fake_chat_mod = types.ModuleType("xai_sdk.chat")
    fake_chat_mod.user = fake_user
    fake_chat_mod.text = fake_text
    fake_chat_mod.image = fake_image

    fake_xai_mod = types.ModuleType("xai_sdk")

    return fake_xai_mod, fake_chat_mod, captured


def _make_fake_client(response_text="4"):
    """
    Return a minimal synchronous SDK client stub whose chat.create().sample()
    returns a fake response with the given text.
    """

    class FakeResponse:
        content = response_text

        class usage:
            prompt_tokens = 10
            completion_tokens = 5

    class FakeChat:
        def create(self, **kw):
            return self

        def append(self, msg):
            return self

        def sample(self):
            return FakeResponse()

    class FakeClient:
        chat = FakeChat()

    return FakeClient()


def _make_fake_async_client(response_text="4"):
    """
    Return a minimal asynchronous SDK client stub whose
    chat.create().sample() is awaitable and returns a fake response.
    """

    class FakeResponse:
        content = response_text

        class usage:
            prompt_tokens = 10
            completion_tokens = 5

    class FakeAsyncChat:
        def create(self, **kw):
            return self

        def append(self, msg):
            return self

        async def sample(self):
            return FakeResponse()

    class FakeAsyncClient:
        chat = FakeAsyncChat()

    return FakeAsyncClient()


#####################################
# API key / model name / temperature
#####################################


def test_grok_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch, settings
):
    """
    Explicit ctor api_key must override Settings.GROK_API_KEY, and the client
    should see a plain string, even if Settings stores a SecretStr.
    """
    # Seed env so Settings sees a GROK_API_KEY
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "env-secret-key"
        settings.GROK_MODEL_NAME = "grok-3"
        settings.GROK_COST_PER_INPUT_TOKEN = 1e-6
        settings.GROK_COST_PER_OUTPUT_TOKEN = 1e-6

    # Sanity: Settings should expose this as SecretStr
    assert isinstance(settings.GROK_API_KEY, SecretStr)

    # Prevent __init__ from importing xai_sdk
    _stub_load_model(monkeypatch)

    # ctor api_key should win over Settings.GROK_API_KEY
    model = GrokModel(
        model="grok-3",
        api_key="ctor-secret-key",
    )

    # _build_client should unwrap the SecretStr to a plain string
    client = model._build_client(_RecordingClient)
    api_key = client.kwargs.get("api_key")

    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


def test_grok_model_defaults_from_settings(monkeypatch, settings):
    """
    When no ctor args are provided, GrokModel should pull model/api_key from
    Settings, which are backed by env vars.
    """
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "env-secret-key"
        settings.GROK_MODEL_NAME = "grok-3"
        settings.GROK_COST_PER_INPUT_TOKEN = 1e-6
        settings.GROK_COST_PER_OUTPUT_TOKEN = 1e-6

    assert isinstance(settings.GROK_API_KEY, SecretStr)

    _stub_load_model(monkeypatch)

    # No ctor args: everything should come from Settings
    model = GrokModel()

    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Client sees the env/Settings value
    assert kw.get("api_key") == "env-secret-key"
    # Model name from Settings
    assert model.name == "grok-3"


##############################
# calculate_cost unit tests  #
##############################


def test_grok_calculate_cost_returns_correct_value(monkeypatch, settings):
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "test-key"
        settings.GROK_COST_PER_INPUT_TOKEN = 0.005
        settings.GROK_COST_PER_OUTPUT_TOKEN = 0.015

    _stub_load_model(monkeypatch)

    model = GrokModel(model="grok-3")
    cost = model.calculate_cost(input_tokens=400, output_tokens=200)
    expected = 400 * 0.005 + 200 * 0.015
    assert cost == expected


def test_grok_calculate_cost_returns_none_when_prices_missing(
    monkeypatch, settings
):
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "test-key"
        settings.GROK_COST_PER_INPUT_TOKEN = 1e-6
        settings.GROK_COST_PER_OUTPUT_TOKEN = 1e-6

    _stub_load_model(monkeypatch)

    model = GrokModel(model="grok-3")
    model.model_data.input_price = None
    model.model_data.output_price = None

    cost = model.calculate_cost(input_tokens=400, output_tokens=200)
    assert cost is None


def test_grok_calculate_cost_with_zero_tokens(monkeypatch, settings):
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "test-key"
        settings.GROK_COST_PER_INPUT_TOKEN = 0.005
        settings.GROK_COST_PER_OUTPUT_TOKEN = 0.015

    _stub_load_model(monkeypatch)

    model = GrokModel(model="grok-3")
    cost = model.calculate_cost(input_tokens=0, output_tokens=0)
    assert cost == 0.0


######################################
# generate / a_generate message shape
######################################


def test_generate_passes_string_to_user(monkeypatch, settings):
    """
    generate() must call xai_sdk.chat.user(str), not user(list-of-dicts).

    Before the fix, content was built as [{"type": "text", "text": prompt}]
    and passed as user(content), which protobuf rejects with TypeError.
    After the fix, user(prompt) is called with the plain string.
    """
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "test-key"
        settings.GROK_COST_PER_INPUT_TOKEN = 1e-6
        settings.GROK_COST_PER_OUTPUT_TOKEN = 1e-6

    fake_xai_mod, fake_chat_mod, captured = _make_fake_xai_sdk()
    sys.modules.setdefault("xai_sdk", fake_xai_mod)
    sys.modules["xai_sdk.chat"] = fake_chat_mod

    fake_client = _make_fake_client("4")

    # Stub load_model at the class level before construction so __init__ does
    # not attempt to import the real xai_sdk Client/AsyncClient.
    _stub_load_model(monkeypatch)

    model = GrokModel(model="grok-3", api_key="test-key")

    # Override on the instance so generate() gets our recording client.
    monkeypatch.setattr(
        model, "load_model", lambda async_mode=False: fake_client
    )

    result, _ = model.generate("What is 2+2?")

    assert result == "4"
    # user() must have been called with exactly one positional arg: the prompt string
    assert "args" in captured, "xai_sdk.chat.user() was never called"
    assert len(captured["args"]) == 1
    assert isinstance(captured["args"][0], str), (
        f"user() received {type(captured['args'][0])!r} instead of str; "
        "list-of-dicts would cause a protobuf TypeError"
    )
    assert captured["args"][0] == "What is 2+2?"


def test_a_generate_passes_string_to_user(monkeypatch, settings):
    """
    a_generate() must call xai_sdk.chat.user(str), not user(list-of-dicts).
    """
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "test-key"
        settings.GROK_COST_PER_INPUT_TOKEN = 1e-6
        settings.GROK_COST_PER_OUTPUT_TOKEN = 1e-6

    fake_xai_mod, fake_chat_mod, captured = _make_fake_xai_sdk()
    sys.modules.setdefault("xai_sdk", fake_xai_mod)
    sys.modules["xai_sdk.chat"] = fake_chat_mod

    fake_client = _make_fake_async_client("4")

    _stub_load_model(monkeypatch)

    model = GrokModel(model="grok-3", api_key="test-key")

    monkeypatch.setattr(
        model, "load_model", lambda async_mode=False: fake_client
    )

    result, _ = asyncio.run(model.a_generate("What is 2+2?"))

    assert result == "4"
    assert "args" in captured, "xai_sdk.chat.user() was never called"
    assert len(captured["args"]) == 1
    assert isinstance(
        captured["args"][0], str
    ), f"user() received {type(captured['args'][0])!r} instead of str"
    assert captured["args"][0] == "What is 2+2?"
