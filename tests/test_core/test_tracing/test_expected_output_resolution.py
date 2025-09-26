import pytest

from deepeval.contextvars import set_current_golden, reset_current_golden
from deepeval.tracing import context as tracing_ctx, observe
from deepeval.utils import is_missing
from deepeval.test_case import LLMTestCase


@pytest.fixture(autouse=True)
def _silence_confident_trace(monkeypatch):
    # quiet logging and don't flush at process end
    monkeypatch.setenv("CONFIDENT_TRACE_VERBOSE", "0")
    monkeypatch.setenv("CONFIDENT_TRACE_FLUSH", "0")

    # patch the real trace_manager object inside its module
    from deepeval.tracing.tracing import trace_manager

    # no-op any network posts
    monkeypatch.setattr(
        trace_manager, "post_trace", lambda *a, **k: None, raising=True
    )

    yield

    from deepeval.tracing.context import (
        current_span_context,
        current_trace_context,
    )

    # reset contextvars
    current_span_context.set(None)
    current_trace_context.set(None)

    # detect leaks, then clear, then fail.
    # This will prevent the scenario where a fail before clear leaves a dirty state for subsequent tests.
    leaked_spans = list(trace_manager.active_spans.keys())
    leaked_traces = list(trace_manager.active_traces.keys())

    # always clear regardless
    trace_manager.active_spans.clear()
    trace_manager.active_traces.clear()

    if leaked_spans or leaked_traces:
        pytest.fail(
            f"Leaked state between tests: spans={leaked_spans}, traces={leaked_traces}"
        )


class _GoldenStub:
    def __init__(self, input=None, expected_output=None):
        self.input = input
        self.expected_output = expected_output


def test_is_missing_helper_covers_whitespace():
    assert is_missing(None) is True
    assert is_missing("") is True
    assert is_missing("   ") is True
    assert is_missing("\n\t ") is True
    assert is_missing("x") is False
    assert is_missing("  x  ") is False


def test_resolve_expected_output_from_context_when_missing():
    # active golden has an expected_output
    tok = set_current_golden(
        _GoldenStub(expected_output="EXPECTED_FROM_DATASET")
    )
    try:
        # missing should resolve from CURRENT_GOLDEN
        assert (
            tracing_ctx._resolve_expected_output_from_context(None)
            == "EXPECTED_FROM_DATASET"
        )
        assert (
            tracing_ctx._resolve_expected_output_from_context("")
            == "EXPECTED_FROM_DATASET"
        )
        assert (
            tracing_ctx._resolve_expected_output_from_context("   ")
            == "EXPECTED_FROM_DATASET"
        )
    finally:
        reset_current_golden(tok)


def test_resolve_expected_output_from_context_respects_explicit_value():
    tok = set_current_golden(
        _GoldenStub(expected_output="EXPECTED_FROM_DATASET")
    )
    try:
        # non-missing should pass through unchanged
        assert (
            tracing_ctx._resolve_expected_output_from_context("USER_VALUE")
            == "USER_VALUE"
        )
        assert (
            tracing_ctx._resolve_expected_output_from_context("  USER_VALUE  ")
            == "  USER_VALUE  "
        )
    finally:
        reset_current_golden(tok)


def test_resolve_expected_output_from_context_when_no_golden_set():
    # no CURRENT_GOLDEN will resolve to the original value. Which is missing in this case
    assert tracing_ctx._resolve_expected_output_from_context(None) is None
    assert tracing_ctx._resolve_expected_output_from_context("") == ""


def test_inherits_expected_output_using_observer_args_with_test_case():
    tok = set_current_golden(
        _GoldenStub(input="china", expected_output="beijing, 1000")
    )
    try:

        @observe(type="llm")
        def tool(input: str):
            # user supplies test_case with input
            tracing_ctx.update_current_span(
                test_case=LLMTestCase(input=input, actual_output="ok")
            )
            # assert while span is still active
            from deepeval.tracing.context import current_span_context

            span = current_span_context.get()
            assert span is not None
            assert span.expected_output == "beijing, 1000"
            return "ok"

        tool("china")
    finally:
        reset_current_golden(tok)


def test_span_inherits_expected_output_using_observer_args_when_test_case_input_omitted():
    tok = set_current_golden(
        _GoldenStub(input="china", expected_output="beijing, 1000")
    )
    try:

        @observe(type="llm")
        def tool(input: str):
            # No test_case input; resolver should use Observer _function_kwargs["input"]
            tracing_ctx.update_current_span(output="ok")

            # Assert while the span is active
            from deepeval.tracing.context import current_span_context

            span = current_span_context.get()
            assert span is not None
            assert span.expected_output == "beijing, 1000"
            return "ok"

        tool("china")
    finally:
        reset_current_golden(tok)


def test_trace_inherits_expected_output_using_observer_args_when_test_case_input_omitted():
    tok = set_current_golden(
        _GoldenStub(input="china", expected_output="beijing, 1000")
    )
    try:

        @observe(type="llm")
        def tool(input: str):
            tracing_ctx.update_current_trace(output="ok")
            from deepeval.tracing.context import current_trace_context

            trace = current_trace_context.get()
            assert trace is not None
            assert trace.expected_output == "beijing, 1000"
            return "ok"

        tool("china")
    finally:
        reset_current_golden(tok)


def test_span_inherits_expected_output_when_input_kwarg_passed_directly():
    tok = set_current_golden(
        _GoldenStub(input="china", expected_output="beijing, 1000")
    )
    try:

        @observe(type="llm")
        def tool():
            # No test_case, instead pass `input` directly via update_current_span
            tracing_ctx.update_current_span(input="china", output="ok")
            from deepeval.tracing.context import current_span_context

            span = current_span_context.get()
            assert span is not None
            assert span.expected_output == "beijing, 1000"
            return "ok"

        tool()
    finally:
        reset_current_golden(tok)


def test_trace_inherits_expected_output_when_input_kwarg_passed_directly():
    tok = set_current_golden(
        _GoldenStub(input="china", expected_output="beijing, 1000")
    )
    try:

        @observe(type="llm")
        def tool():
            # No test_case, instead pass `input` directly via update_current_trace
            tracing_ctx.update_current_trace(input="china", output="ok")
            from deepeval.tracing.context import current_trace_context

            trace = current_trace_context.get()
            assert trace is not None
            assert trace.expected_output == "beijing, 1000"
            return "ok"

        tool()
    finally:
        reset_current_golden(tok)
