import asyncio
import pytest

from deepeval.contextvars import (
    get_current_golden,
)
from deepeval.dataset.golden import Golden
from deepeval.evaluate.execute import (
    a_execute_agentic_test_cases_from_loop,
    execute_agentic_test_cases_from_loop,
)
from deepeval.tracing.context import update_current_span, update_current_trace
from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.evaluate.configs import (
    AsyncConfig,
    DisplayConfig,
    CacheConfig,
    ErrorConfig,
)
from deepeval.evaluate import execute as exec_mod
from .helpers import get_active_trace_and_span


@pytest.fixture(autouse=True)
def _silence_confident_trace(monkeypatch):
    # don’t try to flush leftover traces at process end
    monkeypatch.setenv("CONFIDENT_TRACE_FLUSH", "0")

    # no-op network calls
    monkeypatch.setattr(
        trace_manager, "post_trace", lambda *a, **k: None, raising=True
    )


@pytest.fixture(autouse=True)
def _reset_eval_state():
    yield
    trace_manager.traces_to_evaluate_order.clear()
    trace_manager.traces_to_evaluate.clear()
    trace_manager.integration_traces_to_evaluate.clear()
    trace_manager.test_case_metrics.clear()
    trace_manager.trace_uuid_to_golden.clear()


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

    # simulate user code: create a child span & trace and set actual_output,
    # explicitly passing expected_output from the CURRENT_GOLDEN.
    with Observer("llm", func_name="user"):
        current_golden = get_current_golden()
        update_current_span(
            test_case=LLMTestCase(
                input="china",
                actual_output="beijing, 900",
                expected_output=current_golden.expected_output,
            )
        )
        # executor reads from current_trace, not the span
        update_current_trace(
            test_case=LLMTestCase(
                input="china",
                actual_output="beijing, 900",
                expected_output=current_golden.expected_output,
            )
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
        trace, _ = get_active_trace_and_span()
        assert trace.expected_output == "tc_exp"


def test_trace_kwarg_expected_output_overrides_test_case():
    with Observer("llm", func_name="t2"):
        # test_case provides one value
        update_current_trace(
            test_case=LLMTestCase(
                input="x", actual_output="y", expected_output="tc_exp"
            )
        )
        # but explicit kwarg should win
        update_current_trace(expected_output="kw_exp")
        trace, _ = get_active_trace_and_span()
        assert trace.expected_output == "kw_exp"


def test_trace_expected_output_remains_none_when_unset():
    with Observer("llm", func_name="t4"):
        update_current_trace(
            test_case=LLMTestCase(
                input="x", actual_output="y", expected_output=None
            )
        )
        trace, _ = get_active_trace_and_span()
        assert trace.expected_output is None


def test_span_kwarg_expected_output_overrides_test_case():

    with Observer("llm", func_name="s1"):
        # first set from test_case
        update_current_span(
            test_case=LLMTestCase(
                input="x", actual_output="y", expected_output="from_testcase"
            )
        )
        _, span = get_active_trace_and_span()
        assert span.expected_output == "from_testcase"

        # now explicit kwarg should override
        update_current_span(expected_output="span_kw")
        _, span = get_active_trace_and_span()
        assert span.expected_output == "span_kw"


def test_span_expected_output_remains_none_when_unset():
    with Observer("llm", func_name="s2"):
        update_current_span(
            test_case=LLMTestCase(
                input="x", actual_output="y", expected_output=None
            )
        )
        _, span = get_active_trace_and_span()
        assert span.expected_output is None


def test_noop_when_no_active_trace_or_span():
    # no Observer context -> no current span/trace.
    # these should not crash
    update_current_trace(test_case=LLMTestCase(input="x", actual_output="y"))
    update_current_span(test_case=LLMTestCase(input="x", actual_output="y"))
    # nothing to assert! success == no exception


def test_async_evaluator_skips_empty_traces_without_crash():
    goldens = [Golden(input="x")]
    loop = asyncio.new_event_loop()

    try:
        asyncio.set_event_loop(loop)

        gen = a_execute_agentic_test_cases_from_loop(
            goldens=goldens,
            trace_metrics=None,
            test_results=[],
            loop=loop,
            display_config=DisplayConfig(
                show_indicator=False, verbose_mode=False
            ),
            cache_config=CacheConfig(write_cache=False),
            error_config=ErrorConfig(
                ignore_errors=False, skip_on_missing_params=False
            ),
            async_config=AsyncConfig(run_async=True),
            _use_bar_indicator=False,
        )

        next(gen)

        async def make_empty_traces(n):
            for _ in range(n):
                t = trace_manager.start_new_trace()
                trace_manager.end_trace(t.uuid)  # no spans means empty trace
                await asyncio.sleep(0)

        loop.run_until_complete(make_empty_traces(2))

        with pytest.raises(StopIteration):
            next(gen)

    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_async_evaluator_handles_extra_traces_with_spans():
    goldens = [Golden(input="x")]
    loop = asyncio.new_event_loop()

    try:
        asyncio.set_event_loop(loop)

        gen = a_execute_agentic_test_cases_from_loop(
            goldens=goldens,
            trace_metrics=None,
            test_results=[],
            loop=loop,
            display_config=DisplayConfig(
                show_indicator=False, verbose_mode=False
            ),
            cache_config=CacheConfig(write_cache=False),
            error_config=ErrorConfig(
                ignore_errors=False, skip_on_missing_params=False
            ),
            async_config=AsyncConfig(run_async=True),
            _use_bar_indicator=False,
        )

        next(gen)

        async def make_traces_with_spans(n):
            for _ in range(n):
                # creates a trace and one root span, then closes it
                with Observer("llm", func_name="dummy"):
                    pass
                await asyncio.sleep(0)

        loop.run_until_complete(make_traces_with_spans(2))

        with pytest.raises(StopIteration):
            next(gen)
    finally:
        asyncio.set_event_loop(None)
        loop.close()
