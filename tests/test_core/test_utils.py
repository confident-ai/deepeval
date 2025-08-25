from tenacity.wait import wait_exponential_jitter
from tenacity.stop import stop_after_attempt
from deepeval.utils import read_env_int, read_env_float
from deepeval.models.retry_policy import default_wait, default_stop


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


def test_default_stop_env_override(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_RETRY_MAX_ATTEMPTS", "3")
    stop = default_stop()
    assert isinstance(stop, stop_after_attempt)
    assert stop.max_attempt_number == 3


def test_default_wait_env_override(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_RETRY_INITIAL_SECONDS", "0.5")
    monkeypatch.setenv("DEEPEVAL_RETRY_EXP_BASE", "3")
    monkeypatch.setenv("DEEPEVAL_RETRY_JITTER", "1.5")
    monkeypatch.setenv("DEEPEVAL_RETRY_CAP_SECONDS", "9")

    w = default_wait()
    assert isinstance(w, wait_exponential_jitter)
    # Attributes exposed by tenacity's wait_exponential_jitter:
    assert w.initial == 0.5
    assert w.exp_base == 3
    assert w.jitter == 1.5
    assert w.max == 9


def test_default_wait_ignores_invalid_env(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_RETRY_INITIAL_SECONDS", "nope")
    monkeypatch.setenv("DEEPEVAL_RETRY_EXP_BASE", "NaN")
    monkeypatch.setenv("DEEPEVAL_RETRY_JITTER", "bad")
    monkeypatch.setenv("DEEPEVAL_RETRY_CAP_SECONDS", "-1")
    w = default_wait()
    # falls back to defaults (1, 2, 2, 5)
    assert (
        w.initial == 1.0
        and w.exp_base == 2.0
        and w.jitter == 2.0
        and w.max == 5.0
    )


def test_default_stop_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_RETRY_MAX_ATTEMPTS", "zero?")
    s = default_stop()
    assert isinstance(s, stop_after_attempt) and s.max_attempt_number == 2
