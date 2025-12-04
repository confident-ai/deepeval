import pytest
import openai
import httpx
from tenacity import RetryError
from deepeval.models.llms.openai_model import GPTModel


class AlwaysLengthLimitClient:
    """Fake client that always raises LengthFinishReasonError in the parse path."""

    class _Beta:
        class _Chat:
            class _Completions:
                def __init__(self, counter):
                    self._counter = counter

                def parse(self, *a, **kw):
                    self._counter["calls"] += 1
                    # Raise the (monkeypatched) error class.
                    raise openai.LengthFinishReasonError()

            def __init__(self, counter):
                self.completions = self._Completions(counter)

        def __init__(self, counter):
            self.chat = self._Chat(counter)

    def __init__(self, counter):
        self.beta = self._Beta(counter)


class AlwaysRetryableClient:
    def __init__(self, counter):
        self._counter = counter
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self.chat.completions.create = self._raise

    def _raise(self, *a, **kw):
        self._counter["calls"] += 1
        req = httpx.Request("POST", "https://api.openai.com/v1/fake")
        resp = httpx.Response(
            429, request=req, json={"error": {"code": "rate_limit"}}
        )
        body = {"error": {"code": "rate_limit"}}
        raise openai.RateLimitError(
            message="simulated retryable 429", response=resp, body=body
        )


@pytest.fixture
def gpt_model_retryable(monkeypatch):
    counter = {"calls": 0}

    def _fake_loader(self, async_mode=False):
        return AlwaysRetryableClient(counter)

    monkeypatch.setattr(GPTModel, "load_model", _fake_loader, raising=True)
    return GPTModel(model="gpt-4o-mini"), counter


@pytest.fixture
def gpt_model_length_limit(monkeypatch, settings):
    # Use a local dummy class to stand in for the SDK error (keeps test stable across SDK versions).
    class DummyLengthFinishReasonError(Exception):
        pass

    # Make the name openai.LengthFinishReasonError refer to our dummy class.
    monkeypatch.setattr(
        openai,
        "LengthFinishReasonError",
        DummyLengthFinishReasonError,
        raising=False,
    )

    # Make model use structured outputs path by passing a schema later
    from pydantic import BaseModel

    class DummySchema(BaseModel):
        x: int

    counter = {"calls": 0, "schema": DummySchema}

    def _fake_loader(self, async_mode=False):
        return AlwaysLengthLimitClient(counter)

    with settings.edit(persist=False):
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 5
        settings.DEEPEVAL_RETRY_CAP_SECONDS = 0

    monkeypatch.setattr(GPTModel, "load_model", _fake_loader, raising=True)
    return GPTModel(model="gpt-4o-mini"), counter


def test_retry_respects_max_attempts(
    monkeypatch, gpt_model_retryable, settings
):
    with settings.edit(persist=False):
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 4

    gpt, counter = gpt_model_retryable

    with pytest.raises(RetryError) as excinfo:
        gpt.generate("hello world")

    assert counter["calls"] == 4  # 1 initial + 3 retries
    assert isinstance(
        excinfo.value.last_attempt.exception(), openai.RateLimitError
    )


def test_length_limit_is_non_retryable(gpt_model_length_limit):
    gpt, counter = gpt_model_length_limit
    with pytest.raises(openai.LengthFinishReasonError):
        gpt.generate("any prompt", schema=counter["schema"])
    assert counter["calls"] == 1  # no retries
