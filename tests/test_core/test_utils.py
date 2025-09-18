import pytest
from tenacity import Retrying, wait_fixed, retry_if_exception_type
from tenacity.wait import wait_base
from tenacity.stop import stop_after_attempt, stop_base
from deepeval.utils import read_env_int, read_env_float
from deepeval.models.retry_policy import dynamic_wait, dynamic_stop


def test_read_env_int_valid(monkeypatch):
    monkeypatch.setenv("X_INT", "7")
    assert read_env_int("X_INT", 3) == 7


def test_read_env_int_invalid(monkeypatch):
    monkeypatch.setenv("X_INT", "nope")
    assert read_env_int("X_INT", 3) == 3


def test_read_env_int_min(monkeypatch):
    monkeypatch.setenv("X_INT", "1")
    assert read_env_int("X_INT", 5, min_value=3) == 5


def test_read_env_float_valid(monkeypatch):
    monkeypatch.setenv("X_FLOAT", "2.5")
    assert read_env_float("X_FLOAT", 1.0) == 2.5


def test_read_env_float_invalid(monkeypatch):
    monkeypatch.setenv("X_FLOAT", "nah")
    assert read_env_float("X_FLOAT", 1.0) == 1.0


def test_read_env_float_min(monkeypatch):
    monkeypatch.setenv("X_FLOAT", "0.1")
    assert read_env_float("X_FLOAT", 2.0, min_value=0.5) == 2.0


def test_dynamic_stop_env_override(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_RETRY_MAX_ATTEMPTS", "3")
    stopper = dynamic_stop()

    # It's our own strategy (subclass of stop_base), not stop_after_attempt
    assert isinstance(stopper, stop_base)

    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("boom")

    r = Retrying(
        stop=stopper,
        wait=wait_fixed(0),
        retry=retry_if_exception_type(ValueError),
        reraise=True,
    )

    with pytest.raises(ValueError):
        for attempt in r:
            with attempt:
                boom()

    # 3 total attempts = 1 initial + 2 retries
    assert calls["n"] == 3


def test_dynamic_wait_env_override(monkeypatch):
    # Deterministic (no jitter) + custom params
    monkeypatch.setenv("DEEPEVAL_RETRY_INITIAL_SECONDS", "0.5")
    monkeypatch.setenv("DEEPEVAL_RETRY_EXP_BASE", "3")
    monkeypatch.setenv("DEEPEVAL_RETRY_JITTER", "0")
    monkeypatch.setenv("DEEPEVAL_RETRY_CAP_SECONDS", "9")

    w = dynamic_wait()
    assert isinstance(w, wait_base)  # return a Tenacity wait strategy

    # Record sleeps Tenacity requests between attempts
    sleeps = []

    def fake_sleep(seconds: float):
        sleeps.append(seconds)

    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("boom")

    r = Retrying(
        stop=stop_after_attempt(4),  # total attempts = 4
        wait=w,  # dynamic wait from env
        retry=retry_if_exception_type(
            ValueError
        ),  # keep retrying on ValueError
        reraise=True,
        sleep=fake_sleep,  # capture computed delays
    )

    with pytest.raises(ValueError):
        r(boom)

    # With initial=0.5, base=3, jitter=0, cap=9:
    # waits between attempts:
    # 1 -> [wait] -> 2, 2 -> [wait] -> 3, 3 -> [wait] -> 4
    # should be: 0.5, 1.5, 4.5
    assert sleeps == [0.5, 1.5, 4.5]
    assert calls["n"] == 4  # attempted exactly 4 times


def test_dynamic_stop_invalid_env_falls_back(monkeypatch):
    # Invalid env should fall back to default attempts=2
    monkeypatch.setenv("DEEPEVAL_RETRY_MAX_ATTEMPTS", "zero?")
    stopper = dynamic_stop()
    assert isinstance(stopper, stop_base)

    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("boom")

    r = Retrying(
        stop=stopper,
        wait=wait_fixed(0),
        retry=retry_if_exception_type(ValueError),
        reraise=True,
    )

    with pytest.raises(ValueError):
        r(boom)

    # default attempts == 2 means 1 initial try + 1 retry
    assert calls["n"] == 2
