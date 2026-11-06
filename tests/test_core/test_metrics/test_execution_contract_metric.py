import asyncio
import json

import pytest

from deepeval.test_case import LLMTestCase
from deepeval.tracing.types import ToolSpan, Trace, TraceSpanStatus
from deepeval.evaluate.execute._common import _attach_span_tree
from deepeval.metrics.community.execution_contract.rules import (
    Contract,
    Occurrence,
)
from deepeval.metrics.community.execution_contract.selectors import Selector


def case(*, complete=True, status=TraceSpanStatus.SUCCESS, name="verify"):
    span = ToolSpan(
        uuid="tool",
        trace_uuid="trace",
        name=name,
        start_time=1,
        end_time=2,
        status=status,
        input={"secret": "sensitive payload"},
    )
    trace = Trace(
        uuid="trace",
        start_time=0,
        end_time=3 if complete else None,
        status=status,
        root_spans=[span],
    )
    test_case = LLMTestCase(input="same input")
    _attach_span_tree(test_case, trace, None)
    return test_case


def metric(**kwargs):
    from deepeval.metrics.community import ExecutionContractMetric

    return ExecutionContractMetric(
        Contract(
            (Occurrence("verification", Selector(name="verify"), maximum=1),)
        ),
        **kwargs,
    )


def test_metric_accepts_native_evidence_and_reports_without_payload():
    evaluator = metric()
    assert evaluator.measure(case()) == 1
    assert evaluator.is_successful() is True
    assert evaluator.evaluation_cost == 0
    assert evaluator.report.success
    assert "sensitive payload" not in json.dumps(evaluator.report.to_dict())
    assert "sensitive payload" not in evaluator.reason


def test_incomplete_trace_cannot_pass_even_at_zero_threshold():
    evaluator = metric(threshold=0)
    assert evaluator.measure(case(complete=False)) == 0
    assert evaluator.is_successful() is False
    assert evaluator.report.status.value == "unknown"


def test_reuse_clears_report_score_and_error_before_each_call():
    evaluator = metric()
    evaluator.measure(case())
    with pytest.raises(ValueError, match="native.*evidence"):
        evaluator.measure(LLMTestCase(input="no evidence"))
    assert evaluator.report is None
    assert evaluator.score is None
    assert evaluator.is_successful() is False
    assert evaluator.error
    assert evaluator.measure(case()) == 1
    assert evaluator.error is None


def test_async_and_sync_reports_agree():
    evaluator = metric()
    evaluator.measure(case())
    report = evaluator.report.to_dict()
    assert asyncio.run(evaluator.a_measure(case())) == 1
    assert evaluator.report.to_dict() == report


def test_strict_mode_zeros_partial_score_and_unknown_never_passes():
    from deepeval.metrics.community import ExecutionContractMetric

    contract = Contract(
        (
            Occurrence("verify", Selector(name="verify")),
            Occurrence("pay", Selector(name="pay")),
        )
    )
    relaxed = ExecutionContractMetric(contract, threshold=0.5)
    assert relaxed.measure(case()) == 0.5
    assert relaxed.is_successful()
    strict = ExecutionContractMetric(contract, strict_mode=True, threshold=0.5)
    assert strict.measure(case()) == 0
    assert not strict.is_successful()


@pytest.mark.parametrize("threshold", [True, -1, 2, float("nan"), float("inf")])
def test_threshold_validation(threshold):
    with pytest.raises(ValueError, match="threshold"):
        metric(threshold=threshold)


def test_attach_preserves_legacy_prompt_trace_and_native_completeness():
    test_case = case(complete=False)
    assert test_case._trace_capture_complete is False
    assert test_case._trace_dict is None
    assert case()._trace_capture_complete is True
    assert case(status=TraceSpanStatus.ERRORED)._trace_capture_complete is True


def test_execution_contract_cache_read_is_disabled_even_for_matching_config():
    from deepeval.test_run.cache import Cache, CachedMetricData, CachedTestCase
    from deepeval.evaluate.utils import create_metric_data

    evaluator = metric()
    evaluator.measure(case())
    cached = CachedTestCase(
        cached_metrics_data=[
            CachedMetricData(
                metric_data=create_metric_data(evaluator),
                metric_configuration=Cache.create_metric_configuration(
                    evaluator
                ),
            )
        ]
    )
    assert Cache.get_metric_data(evaluator, cached) is None


def test_native_replay_capture_is_explicit_and_cleared_on_reuse(tmp_path):
    from deepeval.metrics.community.execution_contract.artifacts import (
        write_artifact,
        read_artifact,
    )

    private = metric()
    private.measure(case())
    assert private.evidence_snapshot is None
    captured = metric(capture_evidence=True)
    captured.measure(case())
    snapshot = captured.evidence_snapshot
    assert snapshot.fingerprint == captured.report.evidence_fingerprint
    path = tmp_path / "trace.json"
    write_artifact(path, snapshot)
    assert read_artifact(path) == snapshot
    with pytest.raises(ValueError):
        captured.measure(LLMTestCase(input="missing evidence"))
    assert captured.evidence_snapshot is None
    assert captured.report is None
