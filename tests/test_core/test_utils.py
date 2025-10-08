import pytest
from types import SimpleNamespace
from tenacity import Retrying, wait_fixed, retry_if_exception_type
from tenacity.wait import wait_base
from tenacity.stop import stop_after_attempt, stop_base
from deepeval.utils import read_env_int, read_env_float, shorten
from deepeval.evaluate.utils import _is_metric_successful
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


def test_dynamic_stop_env_override(monkeypatch, settings):
    with settings.edit(persist=False):
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 3
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


def test_dynamic_wait_env_override(monkeypatch, settings):
    # Deterministic (no jitter) + custom params
    with settings.edit(persist=False):
        settings.DEEPEVAL_RETRY_INITIAL_SECONDS = 0.5
        settings.DEEPEVAL_RETRY_EXP_BASE = 3
        settings.DEEPEVAL_RETRY_JITTER = 0
        settings.DEEPEVAL_RETRY_CAP_SECONDS = 9

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


################
# Test shorten #
################


@pytest.mark.parametrize(
    "text,max_len,expected",
    [
        ("hello", 10, "hello"),  # no truncation
        ("hello", 5, "hello"),  # exact boundary
        ("helloworld", 5, "he..."),  # truncation with default suffix
        ("", 5, ""),  # empty string
        (None, 5, ""),  # None -> ""
    ],
)
def test_shorten_basic(text, max_len, expected):
    assert shorten(text, max_len) == expected


def test_shorten_zero_len():
    assert shorten("abc", 0) == ""


def test_shorten_suffix_longer_than_max():
    # max_len < len(suffix) -> suffix is trimmed
    assert shorten("abcdef", 2, suffix="***") == "**"


def test_shorten_non_string_input():
    assert shorten(12345, 3) == "..."


###############################################
# Test evaluate utils - _is_metric_successful #
###############################################


def md(**kw):
    return SimpleNamespace(**kw)


def test_is_metric_successful_priority_error_over_success():
    assert _is_metric_successful(md(error="boom", success=True)) is False


def test_is_metric_successful_bool():
    assert _is_metric_successful(md(error=None, success=True)) is True
    assert _is_metric_successful(md(error=None, success=False)) is False


def test_is_metric_successful_none_and_missing():
    assert _is_metric_successful(md(error=None, success=None)) is False
    assert _is_metric_successful(md(error=None)) is False  # missing attr


def test_is_metric_successful_numeric_and_string():
    assert _is_metric_successful(md(error=None, success=1)) is True
    assert _is_metric_successful(md(error=None, success=0)) is False
    assert _is_metric_successful(md(error=None, success="true")) is True
    assert _is_metric_successful(md(error=None, success="False")) is False
    assert _is_metric_successful(md(error=None, success="YES")) is True
    assert _is_metric_successful(md(error=None, success="no")) is False
