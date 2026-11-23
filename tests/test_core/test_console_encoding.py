"""Regression tests for #3710.

When stdout/stderr use a legacy encoding (e.g. redirected output on Windows,
where the stream encoding is cp1252, cp949, ...), DeepEval's Rich consoles
must not crash with ``UnicodeEncodeError``, and they must not reconfigure the
application's global streams.
"""

import contextlib
import io
import sys

import pytest
from rich.progress import Progress

from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.utils import make_console

EMOJI_LINE = "✨ latest metric 🚀 ⚠ WARNING 한국어"


def _legacy_stream(encoding: str) -> io.TextIOWrapper:
    """A non-interactive text stream with the given encoding, like a
    redirected stdout on Windows."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, newline="\n")


def _read(stream: io.TextIOWrapper) -> str:
    stream.flush()
    return stream.buffer.getvalue().decode(stream.encoding)


@pytest.mark.parametrize("encoding", ["cp1252", "cp949"])
def test_make_console_replaces_unencodable_characters(monkeypatch, encoding):
    stream = _legacy_stream(encoding)
    monkeypatch.setattr(sys, "stdout", stream)

    make_console().print(EMOJI_LINE)

    out = _read(stream)
    assert "latest metric" in out
    assert "✨" not in out
    assert "?" in out
    if encoding == "cp949":
        # characters the encoding can represent are kept as-is
        assert "한국어" in out


def test_make_console_stderr_replaces_unencodable_characters(monkeypatch):
    stream = _legacy_stream("cp1252")
    monkeypatch.setattr(sys, "stderr", stream)

    make_console(stderr=True).print(EMOJI_LINE)

    assert "latest metric" in _read(stream)


def test_make_console_keeps_characters_on_utf8(monkeypatch):
    stream = _legacy_stream("utf-8")
    monkeypatch.setattr(sys, "stdout", stream)

    make_console().print(EMOJI_LINE)

    assert EMOJI_LINE in _read(stream)


def test_make_console_does_not_reconfigure_global_stream(monkeypatch):
    stream = _legacy_stream("cp1252")
    monkeypatch.setattr(sys, "stdout", stream)

    make_console().print(EMOJI_LINE)

    assert sys.stdout is stream
    assert stream.errors == "strict"


def test_make_console_follows_stdout_redirection():
    # created before the redirect, like DeepEval's module-level consoles
    console = make_console()
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        console.print("hello from deepeval")

    assert "hello from deepeval" in buffer.getvalue()


def test_make_console_inside_progress_redirect(monkeypatch):
    # While a live display runs on a terminal, Rich replaces sys.stdout with
    # a FileProxy that writes back to the console. Output printed during the
    # live display must still go through the encoding-safe stream.
    stream = _legacy_stream("cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    console = make_console(force_terminal=True)

    with Progress(console=console, redirect_stdout=True, transient=True):
        print("✨ printed while a progress bar is live")

    assert "printed while a progress bar is live" in _read(stream)


def test_make_console_interactive_path_unchanged(monkeypatch):
    class _Terminal(io.StringIO):
        encoding = "utf-8"

        def isatty(self) -> bool:
            return True

    terminal = _Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)
    console = make_console()

    assert console.is_terminal
    console.print(EMOJI_LINE)
    assert "✨" in terminal.getvalue()


class _ContainsMetric(BaseMetric):
    """Deterministic metric: no model calls."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs):
        self.score = float(test_case.expected_output in test_case.actual_output)
        self.success = self.score >= self.threshold
        self.reason = "contains check"
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Contains"


@pytest.mark.parametrize("encoding", ["cp1252", "cp949"])
def test_evaluate_with_legacy_encoded_output_does_not_crash(
    monkeypatch, encoding
):
    stdout, stderr = _legacy_stream(encoding), _legacy_stream(encoding)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    result = evaluate(
        test_cases=[
            LLMTestCase(
                input="Capital of Korea?",
                actual_output="Seoul is the capital.",
                expected_output="Seoul",
            )
        ],
        metrics=[_ContainsMetric()],
        # write_cache=False avoids #3700, which is unrelated
        cache_config=CacheConfig(write_cache=False),
    )

    assert result.test_results[0].success
    out = _read(stdout)
    # the results panel ("🚀 DeepEval Evaluation Results") and the test run
    # warnings ("⚠ WARNING") are printed with replacement characters
    assert "DeepEval Evaluation Results" in out
    assert "WARNING" in out
