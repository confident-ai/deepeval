import asyncio
import logging
import pytest
from importlib import import_module
from types import SimpleNamespace

from deepeval.dataset import Golden
from deepeval.evaluate.configs import ErrorConfig, DisplayConfig, AsyncConfig
from deepeval.evaluate.types import TestResult
from deepeval.tracing.tracing import trace_manager, Observer
from tests.test_core.stubs import (
    _DummyMetric,
    _DummyTaskCompletionMetric,
    _FakeSpan,
    _FakeTrace,
)
from tests.test_core.helpers import make_trace_api

# module under test
exec_mod = import_module("deepeval.evaluate.execute")


@pytest.fixture
def patched_api_layer(monkeypatch):
    """
    Patch API-creation / conversion helpers so we can pass in simple fake spans/traces
    without needing the full runtime stack. Also patch test-run update to a no-op.
    """

    def _convert_span(_span):
        return SimpleNamespace(status=None, error=None, metrics_data=[])

    trace_api = make_trace_api()
    monkeypatch.setattr(
        exec_mod, "create_api_trace", lambda **_kwargs: trace_api, raising=True
    )
    monkeypatch.setattr(
        trace_manager,
        "_convert_span_to_api_span",
        _convert_span,
        raising=True,
    )
    monkeypatch.setattr(
        trace_manager,
        "create_nested_spans_dict",
        lambda _span: {"dummy": True},
        raising=True,
    )

    # make test_run_manager.update_test_run a no-op
    monkeypatch.setattr(
        exec_mod.global_test_run_manager,
        "update_test_run",
        lambda *_a, **_k: None,
        raising=True,
    )

    # extract_trace_test_results empty by default for these tests
    monkeypatch.setattr(
        exec_mod, "extract_trace_test_results", lambda _api: [], raising=True
    )


@pytest.fixture
def record_measure_calls(monkeypatch):
    """
    Replace measure_metrics_with_indicator with a stub that records which metrics
    were attempted and simulates success (unless metric.skipped was pre-set).
    """
    calls = {"metrics": []}

    async def _stub(metrics, test_case, **_k):
        # emulate the framework's behavior:
        # if a metric has .skipped True already, just leave it
        # otherwise call .measure(), letting metric set .success.
        for m in metrics:
            calls["metrics"].append(m)
            if getattr(m, "skipped", False):
                continue
            # Call the actual metric.measure for our fake metrics
            m.measure(test_case)

    monkeypatch.setattr(
        exec_mod, "measure_metrics_with_indicator", _stub, raising=True
    )
    return calls


#########
# Tests #
#########


@pytest.mark.asyncio
async def test_no_llmtestcase_skips_trace_and_span_metrics(
    patched_api_layer, record_measure_calls
):
    # no input means no trace, so LLMTestCase == None path will trigger.
    trace_metrics = [_DummyMetric(name="trace-metric")]
    span_metrics = [_DummyMetric(name="span-metric")]

    root = _FakeSpan(
        input="span-in", output="span-out", metrics=span_metrics, children=[]
    )
    fake_trace = _FakeTrace(
        input=None, output="trace-out", metrics=trace_metrics, root_span=root
    )

    # run the internal async executor directly to avoid building an observed callback.
    results: list[TestResult] = []
    golden = Golden(input="golden-input")
    await exec_mod._a_execute_agentic_test_case(
        golden=golden,
        test_run_manager=exec_mod.global_test_run_manager,
        test_results=results,
        count=1,
        verbose_mode=False,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False,
        _use_bar_indicator=False,
        _is_assert_test=False,
        observed_callback=None,
        trace=fake_trace,
        trace_metrics=None,  # use the ones on our fake trace
        progress=None,
        pbar_id=None,
    )

    # We expect:
    # - trace-level metric did not get measured do to invalid or missing LLMTestCase
    # - span-level metric did not run
    names_called = {
        getattr(m, "name", "<noname>") for m in record_measure_calls["metrics"]
    }
    assert "span-metric" not in names_called
    assert "trace-metric" not in names_called

    # and a top level TestResult should be produced
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_trace_error_boundary_no_actual_output_still_evaluates_span_metrics(
    patched_api_layer, record_measure_calls
):
    trace_metrics = [_DummyMetric(name="trace-metric")]
    span_metrics = [_DummyMetric(name="span-metric")]

    # input present, but output is None hits the "No actual_output" branch
    root = _FakeSpan(
        input="span-in", output="span-out", metrics=span_metrics, children=[]
    )
    fake_trace = _FakeTrace(
        input="trace-in", output=None, metrics=trace_metrics, root_span=root
    )

    results: list[TestResult] = []
    golden = Golden(input="golden-input")
    await exec_mod._a_execute_agentic_test_case(
        golden=golden,
        test_run_manager=exec_mod.global_test_run_manager,
        test_results=results,
        count=1,
        verbose_mode=False,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False,
        _use_bar_indicator=False,
        _is_assert_test=False,
        observed_callback=None,
        trace=fake_trace,
        trace_metrics=None,
        progress=None,
        pbar_id=None,
    )

    names_called = {
        getattr(m, "name", "<noname>") for m in record_measure_calls["metrics"]
    }
    assert "span-metric" in names_called
    assert "trace-metric" in names_called
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_task_completion_path_sets_trace_case_and_evaluates_metrics(
    patched_api_layer, record_measure_calls
):
    """
    For completeness, ensure that when a TaskCompletionMetric is present at trace level,
    trace metrics are executed
    """
    # Include a TaskCompletionMetric so the "has_task_completion" branch is taken
    trace_metrics = [
        _DummyTaskCompletionMetric(name="tc"),
        _DummyMetric(name="trace-metric"),
    ]
    # No span metrics needed here, we just want to see the trace metrics measured
    root = _FakeSpan(
        input="span-in", output="span-out", metrics=[], children=[]
    )
    # Note: if input is present, then output can be None because it is optional.
    fake_trace = _FakeTrace(
        input="trace-in", output=None, metrics=trace_metrics, root_span=root
    )

    results: list[TestResult] = []
    golden = Golden(input="golden-input")
    await exec_mod._a_execute_agentic_test_case(
        golden=golden,
        test_run_manager=exec_mod.global_test_run_manager,
        test_results=results,
        count=1,
        verbose_mode=False,
        ignore_errors=True,
        skip_on_missing_params=True,
        show_indicator=False,
        _use_bar_indicator=False,
        _is_assert_test=False,
        observed_callback=None,
        trace=fake_trace,
        trace_metrics=None,
        progress=None,
        pbar_id=None,
    )

    names_called = {
        getattr(m, "name", "<noname>") for m in record_measure_calls["metrics"]
    }
    # Both the TaskCompletionMetric and the normal trace metric should have been measured
    assert "tc" in names_called
    assert "trace-metric" in names_called
    assert len(results) >= 1


def test_task_exception_logs_error_when_debug_enabled(
    monkeypatch, caplog, settings
):

    with settings.edit(persist=False):
        settings.DEEPEVAL_DEBUG_ASYNC = 1

    # Capture logs from deepeval.evaluate.execute
    caplog.set_level(logging.INFO, logger="deepeval.evaluate.execute")

    # do not expect metrics to run in this scenario
    calls = {"measurements": 0}

    async def _noop_measure(metrics, test_case, **_):
        calls["measurements"] += 1

    monkeypatch.setattr(
        exec_mod, "measure_metrics_with_indicator", _noop_measure, raising=True
    )

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        goldens = [Golden(input="What's the weather like in SF?")]
        results: list[TestResult] = []

        it = exec_mod.a_execute_agentic_test_cases_from_loop(
            goldens=goldens,
            trace_metrics=[_DummyMetric()],
            test_results=results,
            loop=loop,
            display_config=DisplayConfig(show_indicator=False),
            async_config=AsyncConfig(run_async=True),
            error_config=ErrorConfig(
                ignore_errors=True, skip_on_missing_params=True
            ),
        )

        golden = next(it)

        async def failing_app(_):
            raise RuntimeError("Network down / DNS failure")

        task = asyncio.create_task(failing_app(golden.input))
        try:
            it.send(task)  # register the task with the iterator
        except StopIteration:
            pass

        # drain iterator, this runs the task
        for _ in it:
            pass

        assert calls["measurements"] == 0
        assert isinstance(results, list)

        assert not trace_manager.traces_to_evaluate
        assert not trace_manager.integration_traces_to_evaluate

        # An error log should have been emitted by on_task_done
        assert any("task ERROR" in r.message for r in caplog.records)
        assert any(
            "Network down / DNS failure" in (r.exc_text or "")
            for r in caplog.records
        )

    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_task_error_after_observe_marks_existing_trace(monkeypatch):
    from deepeval.tracing.tracing import trace_manager
    from deepeval.tracing.context import current_trace_context
    from deepeval.dataset import Golden
    from deepeval.evaluate.configs import (
        DisplayConfig,
        AsyncConfig,
        ErrorConfig,
    )

    # Don’t execute real metrics
    monkeypatch.setattr(
        exec_mod,
        "measure_metrics_with_indicator",
        lambda *a, **k: None,
        raising=True,
    )

    captured = {"trace": None}

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        goldens = [Golden(input="hi")]
        test_results = []

        it = exec_mod.a_execute_agentic_test_cases_from_loop(
            goldens=goldens,
            trace_metrics=None,
            test_results=test_results,
            loop=loop,
            display_config=DisplayConfig(show_indicator=False),
            async_config=AsyncConfig(run_async=True),
            error_config=ErrorConfig(
                ignore_errors=True, skip_on_missing_params=True
            ),
        )

        golden = next(it)

        async def app(_):
            # create a trace under Observer, then fail after it exists.
            with Observer("custom", func_name="unit-test"):
                trace = current_trace_context.get()
                # make sure on_task_done can find and mark this trace
                trace_manager.trace_uuid_to_golden[trace.uuid] = golden
                if trace not in trace_manager.integration_traces_to_evaluate:
                    trace_manager.integration_traces_to_evaluate.append(trace)
                captured["trace"] = trace
                # fail after observe
                await asyncio.sleep(0)
                raise RuntimeError("boom after observe")

        task = asyncio.create_task(app(golden.input))
        try:
            it.send(
                task
            )  # register with the iterator so it tracks and awaits it
        except StopIteration:
            pass

        # drain the iterator, it will await the task, run on_task_done, then evaluate traces.
        for _ in it:
            pass

        # assert on the concrete trace object
        assert captured["trace"] is not None, "expected a trace to exist"
        tr = captured["trace"]
        assert tr is not None
        assert getattr(tr.status, "name", str(tr.status)) == "ERRORED"

        last = tr.root_spans[-1] if tr.root_spans else None
        err_text = (last.error if last else "") or ""
        assert "boom after observe" in err_text

    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_task_cancel_after_observe_marks_existing_trace(monkeypatch):
    from deepeval.tracing.tracing import trace_manager
    from deepeval.tracing.context import current_trace_context

    try:
        from deepeval.tracing.tracing import update_current_trace
    except Exception:
        update_current_trace = None

    # no real metrics
    monkeypatch.setattr(
        exec_mod,
        "measure_metrics_with_indicator",
        lambda *a, **k: None,
        raising=True,
    )

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        goldens = [Golden(input="hello")]
        results: list[TestResult] = []

        it = exec_mod.a_execute_agentic_test_cases_from_loop(
            goldens=goldens,
            trace_metrics=None,
            test_results=results,
            loop=loop,
            display_config=DisplayConfig(show_indicator=False),
            async_config=AsyncConfig(run_async=True),
            error_config=ErrorConfig(
                ignore_errors=True, skip_on_missing_params=True
            ),
        )

        golden = next(it)

        captured = {"trace": None}

        async def app(_):
            with Observer("custom", func_name="unit-test"):
                if update_current_trace is not None:
                    try:
                        update_current_trace(input="x", output=None)
                    except Exception:
                        pass
                tr = current_trace_context.get()
                captured["trace"] = tr
                trace_manager.trace_uuid_to_golden[tr.uuid] = golden
                if tr not in trace_manager.integration_traces_to_evaluate:
                    trace_manager.integration_traces_to_evaluate.append(tr)

                # yield once so the task actually starts and mapping is in place
                await asyncio.sleep(0)
                # simulate cancellation
                raise asyncio.CancelledError()

        task = asyncio.create_task(app(golden.input))
        try:
            it.send(task)
        except StopIteration:
            pass

        # drain so that the iterator will run the loop, invoke on_task_done, and evaluate traces
        for _ in it:
            pass

        tr = captured["trace"]
        assert tr is not None, "expected a trace to exist"
        assert getattr(tr.status, "name", str(tr.status)) == "ERRORED"
        # last root span should carry a cancel message
        assert tr.root_spans and tr.root_spans[-1].error
        assert "cancelled" in tr.root_spans[-1].error.lower()
    finally:
        asyncio.set_event_loop(None)
        loop.close()
