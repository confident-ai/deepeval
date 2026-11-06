import importlib

from deepeval.dataset import Golden
from deepeval.evaluate.configs import CacheConfig, DisplayConfig
from deepeval.evaluate.execute.loop import execute_agentic_test_cases_from_loop
from deepeval.metrics.community.execution_contract import (
    Contract,
    ExecutionContractMetric,
    Occurrence,
    Selector,
)
from deepeval.test_run.test_run import (
    TestRun as Run,
    TestRunManager as RunManager,
)
from deepeval.tracing import observe


def test_sync_iterator_captures_every_top_level_user_call(monkeypatch):
    module = importlib.import_module("deepeval.evaluate.execute.loop")
    manager = RunManager()
    manager.save_to_disk = False
    manager.set_test_run(Run(identifier="scope"))
    monkeypatch.setattr(module, "global_test_run_manager", manager)
    check = ExecutionContractMetric(
        Contract(
            (
                Occurrence(
                    "no_payment", Selector(name="pay"), minimum=0, maximum=0
                ),
            )
        )
    )

    @observe(name="verify")
    def verify():
        return True

    @observe(name="pay")
    def pay():
        return "paid"

    for _ in execute_agentic_test_cases_from_loop(
        goldens=[Golden(input="x")],
        trace_metrics=[check],
        test_results=[],
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(write_cache=False),
    ):
        verify()
        pay()
    assert check.report.status.value == "fail"
    assert check.report.rules[0].reason_code == "too_many_occurrences"


def test_trace_scope_excludes_only_framework_wrapper(monkeypatch):
    from deepeval.constants import PYTEST_TRACE_TEST_WRAPPER_SPAN_NAME
    from deepeval.evaluate.execute.trace_scope import (
        _assert_test_from_current_trace,
    )
    from deepeval.tracing.context import current_trace_context
    from deepeval.tracing.types import (
        BaseSpan,
        ToolSpan,
        Trace,
        TraceSpanStatus,
    )

    module = importlib.import_module("deepeval.evaluate.execute.trace_scope")
    manager = RunManager()
    manager.save_to_disk = False
    manager.set_test_run(Run(identifier="scope"))
    monkeypatch.setattr(module, "global_test_run_manager", manager)
    child = ToolSpan(
        uuid="child",
        trace_uuid="trace",
        parent_uuid="wrapper",
        name="verify",
        start_time=1,
        end_time=2,
        status=TraceSpanStatus.SUCCESS,
    )
    wrapper = BaseSpan(
        uuid="wrapper",
        trace_uuid="trace",
        name=PYTEST_TRACE_TEST_WRAPPER_SPAN_NAME,
        start_time=0.1,
        end_time=None,
        status=TraceSpanStatus.IN_PROGRESS,
        children=[child],
    )
    trace = Trace(
        uuid="trace",
        start_time=0.1,
        end_time=None,
        root_spans=[wrapper],
        status=TraceSpanStatus.IN_PROGRESS,
    )
    check = ExecutionContractMetric(
        Contract(
            (Occurrence("no_pay", Selector(name="pay"), minimum=0, maximum=0),)
        )
    )
    token = current_trace_context.set(trace)
    try:
        _assert_test_from_current_trace(Golden(input="x"), metrics=[check])
        assert check.report.status.value == "pass"
        child.status, child.end_time = TraceSpanStatus.IN_PROGRESS, None
        _assert_test_from_current_trace(Golden(input="x"), metrics=[check])
        assert check.report.status.value == "unknown"
    finally:
        current_trace_context.reset(token)


def test_native_only_metric_does_not_build_prompt_trace(monkeypatch):
    from deepeval.tracing.tracing import trace_manager

    module = importlib.import_module("deepeval.evaluate.execute.loop")
    manager = RunManager()
    manager.save_to_disk = False
    manager.set_test_run(Run(identifier="native-only"))
    monkeypatch.setattr(module, "global_test_run_manager", manager)

    def compact_trace_is_not_needed(*args):
        raise AssertionError("native metric requested compact prompt trace")

    monkeypatch.setattr(
        trace_manager, "create_nested_spans_dict", compact_trace_is_not_needed
    )
    check = ExecutionContractMetric(
        Contract((Occurrence("verify", Selector(name="verify")),))
    )

    @observe(name="verify")
    def verify():
        return True

    for _ in execute_agentic_test_cases_from_loop(
        goldens=[Golden(input="x")],
        trace_metrics=[check],
        test_results=[],
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(write_cache=False),
    ):
        verify()
    assert check.report.success


def test_metric_counting_rejects_cycles_without_recursive_failure():
    import pytest
    from deepeval.evaluate.utils import count_metrics_in_trace
    from deepeval.tracing.types import BaseSpan, Trace, TraceSpanStatus

    span = BaseSpan(
        uuid="one",
        trace_uuid="trace",
        name="one",
        start_time=1,
        end_time=2,
        status=TraceSpanStatus.SUCCESS,
    )
    span.children = [span]
    trace = Trace(
        uuid="trace",
        root_spans=[span],
        status=TraceSpanStatus.SUCCESS,
        start_time=1,
        end_time=2,
    )
    with pytest.raises(ValueError, match="cycle|duplicate"):
        count_metrics_in_trace(trace)
