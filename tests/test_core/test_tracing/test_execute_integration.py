import pytest

from deepeval.contextvars import (
    get_current_golden,
    set_current_golden,
    reset_current_golden,
)
from deepeval.dataset.golden import Golden
from deepeval.evaluate.execute import execute_agentic_test_cases_from_loop
from deepeval.tracing.context import update_current_span, update_current_trace
from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.evaluate.configs import DisplayConfig, CacheConfig, ErrorConfig
from deepeval.evaluate import execute as exec_mod


@pytest.fixture(autouse=True)
def _silence_confident_trace(monkeypatch):
    # don’t try to flush leftover traces at process end
    monkeypatch.setenv("CONFIDENT_TRACE_FLUSH", "0")

    # no-op network calls
    monkeypatch.setattr(
        trace_manager, "post_trace", lambda *a, **k: None, raising=True
    )


class GoldenStub:
    def __init__(self, expected_output=None):
        self.expected_output = expected_output


def _get_active_trace_and_span():
    # helper to peek at current trace/span via the observer context
    from deepeval.tracing.context import (
        current_trace_context,
        current_span_context,
    )

    return current_trace_context.get(), current_span_context.get()


def test_execute_propagates_expected_output(monkeypatch):
    received_test_cases = []

    # patch the symbol that execute.py calls
    orig_create_api_test_case = exec_mod.create_api_test_case

    def spy_create_api_test_case(*, test_case, trace, index=None):
        received_test_cases.append(test_case)
        return orig_create_api_test_case(
            test_case=test_case, trace=trace, index=index
        )

    monkeypatch.setattr(
        exec_mod, "create_api_test_case", spy_create_api_test_case
    )

    goldens = [Golden(input="china", expected_output="beijing, 1000")]

    gen = execute_agentic_test_cases_from_loop(
        goldens=goldens,
        trace_metrics=None,
        test_results=[],
        display_config=DisplayConfig(show_indicator=False, verbose_mode=False),
        cache_config=CacheConfig(write_cache=False),
        error_config=ErrorConfig(
            ignore_errors=False, skip_on_missing_params=False
        ),
        _use_bar_indicator=False,
    )

    # The executor yields the current golden first
    golden = next(gen)
    assert golden.input == "china"

    # simulate user code: create a child span and set actual_output,
    # leaving expected_output empty so it’s resolved from CURRENT_GOLDEN.
    with Observer("llm", func_name="user"):
        update_current_span(
            test_case=LLMTestCase(input="china", actual_output="beijing, 900")
        )
        # executor reads from current_trace, not the span
        update_current_trace(
            test_case=LLMTestCase(input="china", actual_output="beijing, 900")
        )

    # resume executor so it builds the test case and hits our spy
    with pytest.raises(StopIteration):
        next(gen)

    assert len(received_test_cases) == 1
    tc = received_test_cases[0]
    assert tc.input == "china"
    assert tc.actual_output == "beijing, 900"
    assert (
        tc.expected_output == "beijing, 1000"
    )  # This should be set via CURRENT_GOLDEN
    assert get_current_golden() is None


def test_trace_uses_test_case_expected_output_when_present():
    with Observer("llm", func_name="t1"):
        update_current_trace(
            test_case=LLMTestCase(
                input="x", actual_output="y", expected_output="tc_exp"
            )
        )
        trace, _ = _get_active_trace_and_span()
        assert trace.expected_output == "tc_exp"


def test_trace_kwarg_expected_output_overrides_test_case_and_golden():
    tok = set_current_golden(GoldenStub(expected_output="golden_exp"))
    try:
        with Observer("llm", func_name="t2"):
            # test_case provides one value
            update_current_trace(
                test_case=LLMTestCase(
                    input="x", actual_output="y", expected_output="tc_exp"
                )
            )
            # but explicit kwarg should win
            update_current_trace(expected_output="kw_exp")
            trace, _ = _get_active_trace_and_span()
            assert trace.expected_output == "kw_exp"
    finally:
        reset_current_golden(tok)


def test_trace_resolves_from_golden_when_missing_or_blank():
    # Golden exists with an expected_output, let the test_case omit it
    tok = set_current_golden(GoldenStub(expected_output="golden_exp"))
    try:
        with Observer("llm", func_name="t3"):
            update_current_trace(
                test_case=LLMTestCase(
                    input="x", actual_output="y", expected_output=None
                )
            )
            trace, _ = _get_active_trace_and_span()
            assert trace.expected_output == "golden_exp"

        # Also cover "blank string" -> treated as missing
        with Observer("llm", func_name="t3b"):
            update_current_trace(
                test_case=LLMTestCase(
                    input="x",
                    actual_output="y",
                    expected_output="   ",  # white space is treated as empty / not set
                )
            )
            trace, _ = _get_active_trace_and_span()
            assert trace.expected_output == "golden_exp"
    finally:
        reset_current_golden(tok)


def test_trace_stays_none_when_missing_and_no_golden():
    with Observer("llm", func_name="t4"):
        update_current_trace(
            test_case=LLMTestCase(
                input="x", actual_output="y", expected_output=None
            )
        )
        trace, _ = _get_active_trace_and_span()
        assert trace.expected_output is None


def test_span_kwarg_expected_output_overrides_everything():
    tok = set_current_golden(GoldenStub(expected_output="golden_exp"))
    try:
        with Observer("llm", func_name="s1"):
            # first set from test_case, which should resolve from golden
            update_current_span(
                test_case=LLMTestCase(
                    input="x", actual_output="y", expected_output=None
                )
            )
            _, span = _get_active_trace_and_span()
            # resolve from golden
            assert span.expected_output == "golden_exp"

            # now explicit kwarg should override
            update_current_span(expected_output="span_kw")
            _, span = _get_active_trace_and_span()
            assert span.expected_output == "span_kw"
    finally:
        reset_current_golden(tok)


def test_span_stays_none_when_missing_and_no_golden():
    with Observer("llm", func_name="s2"):
        update_current_span(
            test_case=LLMTestCase(
                input="x", actual_output="y", expected_output=None
            )
        )
        _, span = _get_active_trace_and_span()
        assert span.expected_output is None


def test_noop_when_no_active_trace_or_span():
    # no Observer context -> no current span/trace.
    # these should not crash
    update_current_trace(test_case=LLMTestCase(input="x", actual_output="y"))
    update_current_span(test_case=LLMTestCase(input="x", actual_output="y"))
    # nothing to assert! success == no exception
