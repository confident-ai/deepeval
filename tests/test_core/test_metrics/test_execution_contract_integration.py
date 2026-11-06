import asyncio
import importlib

import pytest

from deepeval.dataset import Golden
from deepeval.evaluate.configs import CacheConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.execute.agentic import _a_execute_agentic_test_case
from deepeval.metrics.base_metric import BaseMetric
from deepeval.metrics.community import ExecutionContractMetric
from deepeval.metrics.community.execution_contract import (
    Contract,
    Occurrence,
    Selector,
)
from deepeval.test_run.test_run import (
    TestRun as Run,
    TestRunManager as RunManager,
)
from deepeval.tracing.types import AgentSpan, ToolSpan, Trace, TraceSpanStatus


class LegacyMetric(BaseMetric):
    def __init__(self):
        self.threshold = 1
        self.calls = 0

    def measure(self, test_case, *args, **kwargs):
        self.calls += 1
        self.score = 1
        self.success = True
        return 1

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    @property
    def __name__(self):
        return "Legacy"


def contract(name="verify", status=None):
    return ExecutionContractMetric(
        Contract((Occurrence("observed", Selector(name=name, status=status)),))
    )


def manager():
    result = RunManager()
    result.save_to_disk = False
    result.set_test_run(Run(identifier="contracts"))
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_failed", [False, True])
async def test_async_failed_parent_keeps_contracts_and_skips_legacy_descendants(
    trace_failed,
):
    child_contract, root_contract, trace_contract = (
        contract(),
        contract(),
        contract(),
    )
    legacy_child, legacy_root, legacy_trace = (
        LegacyMetric(),
        LegacyMetric(),
        LegacyMetric(),
    )
    child = ToolSpan(
        uuid="child",
        trace_uuid="trace",
        parent_uuid="root",
        name="verify",
        input="x",
        output="ok",
        start_time=1,
        end_time=2,
        status=TraceSpanStatus.SUCCESS,
        metrics=[child_contract, legacy_child],
    )
    root = AgentSpan(
        uuid="root",
        trace_uuid="trace",
        name="agent",
        input="x",
        start_time=0.1,
        end_time=3,
        status=TraceSpanStatus.ERRORED,
        error="application failed",
        children=[child],
        metrics=[root_contract, legacy_root],
    )
    trace = Trace(
        uuid="trace",
        start_time=0.1,
        end_time=4,
        input="x",
        status=(
            TraceSpanStatus.ERRORED if trace_failed else TraceSpanStatus.SUCCESS
        ),
        root_spans=[root],
        metrics=[trace_contract, legacy_trace],
    )
    trm = manager()
    await _a_execute_agentic_test_case(
        golden=Golden(input="x"),
        test_run_manager=trm,
        test_results=[],
        count=0,
        verbose_mode=False,
        ignore_errors=False,
        skip_on_missing_params=False,
        show_indicator=False,
        _use_bar_indicator=False,
        _is_assert_test=False,
        trace=trace,
    )
    for metric in (child_contract, root_contract, trace_contract):
        assert metric.report is not None
        assert metric.report.success
    assert legacy_root.calls == legacy_child.calls == 0
    assert legacy_trace.calls == (0 if trace_failed else 1)
    recorded = trm.get_test_run().test_cases[0].trace
    assert recorded.agent_spans[0].status == "ERRORED"
    assert recorded.agent_spans[0].error == "application failed"
    assert trace.status == (
        TraceSpanStatus.ERRORED if trace_failed else TraceSpanStatus.SUCCESS
    )


def test_sync_iterator_evaluates_failed_component_contract_and_skips_legacy(
    monkeypatch,
):
    from deepeval.evaluate.execute.loop import (
        execute_agentic_test_cases_from_loop,
    )
    from deepeval.tracing import observe

    module = importlib.import_module("deepeval.evaluate.execute.loop")
    trm = manager()
    monkeypatch.setattr(module, "global_test_run_manager", trm)
    check, legacy = contract(name="failing", status="error"), LegacyMetric()

    @observe(name="failing", metrics=[check, legacy])
    def failing(value):
        raise RuntimeError("expected failure")

    for golden in execute_agentic_test_cases_from_loop(
        goldens=[Golden(input="x")],
        trace_metrics=[],
        test_results=[],
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(write_cache=False),
    ):
        try:
            failing(golden.input)
        except RuntimeError:
            pass
    assert check.report is not None
    assert check.report.success
    assert legacy.calls == 0


def test_trace_scope_can_evaluate_an_errored_trace(monkeypatch):
    from deepeval.evaluate.execute.trace_scope import (
        _assert_test_from_current_trace,
    )
    from deepeval.tracing.context import current_trace_context

    module = importlib.import_module("deepeval.evaluate.execute.trace_scope")
    monkeypatch.setattr(module, "global_test_run_manager", manager())
    check, legacy = contract(name="failing", status="error"), LegacyMetric()
    root = AgentSpan(
        uuid="root",
        trace_uuid="trace",
        name="failing",
        start_time=0.1,
        end_time=1,
        status=TraceSpanStatus.ERRORED,
        error="expected failure",
        metrics=[legacy],
    )
    trace = Trace(
        uuid="trace",
        start_time=0.1,
        end_time=2,
        status=TraceSpanStatus.ERRORED,
        root_spans=[root],
    )
    token = current_trace_context.set(trace)
    try:
        result = _assert_test_from_current_trace(
            Golden(input="x"), metrics=[check, legacy]
        )
    finally:
        current_trace_context.reset(token)
    assert check.report is not None
    assert check.report.success
    assert legacy.calls == 0


@pytest.mark.asyncio
async def test_empty_trace_can_check_required_occurrence_without_index_error():
    check = contract()
    trace = Trace(
        uuid="empty",
        start_time=0.1,
        end_time=1,
        status=TraceSpanStatus.ERRORED,
        root_spans=[],
        metrics=[check],
    )
    await _a_execute_agentic_test_case(
        golden=Golden(input="x"),
        test_run_manager=manager(),
        test_results=[],
        count=0,
        verbose_mode=False,
        ignore_errors=False,
        skip_on_missing_params=False,
        show_indicator=False,
        _use_bar_indicator=False,
        _is_assert_test=False,
        trace=trace,
    )
    assert check.report is not None
    assert check.report.status.value == "fail"


@pytest.mark.parametrize("run_async", [False, True])
def test_e2e_cache_never_reads_or_writes_contract_results(
    monkeypatch, run_async
):
    from deepeval.evaluate.execute.e2e import (
        execute_test_cases,
        _a_execute_llm_test_cases,
    )
    from deepeval.evaluate.execute._common import _attach_span_tree
    from deepeval.evaluate.utils import create_metric_data
    from deepeval.test_case import LLMTestCase
    from deepeval.test_run.cache import Cache, CachedMetricData, CachedTestCase

    module = importlib.import_module("deepeval.evaluate.execute.e2e")
    span = ToolSpan(
        uuid="tool",
        trace_uuid="trace",
        name="verify",
        start_time=1,
        end_time=2,
        status=TraceSpanStatus.SUCCESS,
    )
    trace = Trace(
        uuid="trace",
        start_time=0.1,
        end_time=3,
        root_spans=[span],
        status=TraceSpanStatus.SUCCESS,
    )
    test_case = LLMTestCase(input="same input")
    _attach_span_tree(test_case, trace, None)
    previous = contract("verify")
    previous.measure(test_case)
    # Simulate an old/third-party cache entry with a colliding metric name and
    # the same standard configuration but a different execution contract.
    stale = CachedTestCase(
        cached_metrics_data=[
            CachedMetricData(
                metric_data=create_metric_data(previous),
                metric_configuration=Cache.create_metric_configuration(
                    previous
                ),
            )
        ]
    )
    writes = []
    monkeypatch.setattr(
        module.global_test_run_cache_manager,
        "get_cached_test_case",
        lambda *args, **kwargs: stale,
    )
    monkeypatch.setattr(
        module.global_test_run_cache_manager,
        "cache_test_case",
        lambda tc, cached, *args, **kwargs: writes.append(cached),
    )
    check, legacy = contract("pay"), LegacyMetric()
    trm = manager()
    if run_async:
        asyncio.run(
            _a_execute_llm_test_cases(
                metrics=[check, legacy],
                test_case=test_case,
                test_run_manager=trm,
                test_results=[],
                count=0,
                test_run=trm.get_test_run(),
                ignore_errors=False,
                skip_on_missing_params=False,
                use_cache=True,
                show_indicator=False,
                _use_bar_indicator=False,
                _is_assert_test=False,
            )
        )
    else:
        execute_test_cases(
            test_cases=[test_case],
            metrics=[check, legacy],
            test_run_manager=trm,
            cache_config=CacheConfig(use_cache=True, write_cache=False),
            display_config=DisplayConfig(
                show_indicator=False, print_results=False
            ),
        )
    assert check.report.status.value == "fail"
    assert check.score == 0
    assert not check.is_successful()
    assert previous.report.success
    assert writes
    for cached in writes:
        assert [m.metric_data.name for m in cached.cached_metrics_data] == [
            "Legacy"
        ]
