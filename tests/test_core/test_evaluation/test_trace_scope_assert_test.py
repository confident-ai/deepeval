from time import perf_counter

from deepeval.dataset import Golden
from deepeval.constants import PYTEST_TRACE_TEST_WRAPPER_SPAN_NAME
from deepeval.evaluate.configs import DisplayConfig, ErrorConfig
from deepeval.evaluate.execute import trace_scope as trace_scope_mod
from deepeval.evaluate.execute.trace_scope import _assert_test_from_current_trace
from deepeval.metrics import BaseMetric
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.types import BaseSpan, Trace, TraceSpanStatus
from tests.test_core.stubs import make_span_api_like


class CapturingMetric(BaseMetric):
    def __init__(self, expected_input: str, expected_output: str):
        self.expected_input = expected_input
        self.expected_output = expected_output
        self.threshold = 1.0
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.strict_mode = False
        self.evaluation_model = None
        self.evaluation_cost = None
        self.verbose_logs = None
        self.skipped = False

    @property
    def __name__(self):
        return "CapturingMetric"

    def measure(self, test_case, *args, **kwargs):
        self.reason = f"{test_case.input} -> {test_case.actual_output}"
        self.score = float(
            test_case.input == self.expected_input
            and test_case.actual_output == self.expected_output
        )
        self.success = self.score == 1.0
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self):
        return bool(self.success)


def _make_pytest_wrapped_trace(app_span: BaseSpan) -> Trace:
    now = perf_counter()
    wrapper = BaseSpan(
        uuid="wrapper",
        status=TraceSpanStatus.SUCCESS,
        children=[app_span],
        trace_uuid="trace",
        parent_uuid=None,
        start_time=now,
        end_time=now,
        name=PYTEST_TRACE_TEST_WRAPPER_SPAN_NAME,
    )
    return Trace(
        uuid="trace",
        status=TraceSpanStatus.SUCCESS,
        root_spans=[wrapper],
        start_time=now,
        end_time=None,
    )


def test_assert_test_metrics_run_at_trace_level_with_golden_input(
    monkeypatch,
):
    app_span = BaseSpan(
        uuid="app",
        status=TraceSpanStatus.SUCCESS,
        children=[],
        trace_uuid="trace",
        parent_uuid="wrapper",
        start_time=perf_counter(),
        end_time=perf_counter(),
        name="llm_app",
        input={"query": "ignored for trace metrics"},
        output="trace answer",
    )
    trace = _make_pytest_wrapped_trace(app_span)

    monkeypatch.setattr(
        trace_scope_mod.trace_manager,
        "_convert_span_to_api_span",
        lambda *_: make_span_api_like(),
        raising=True,
    )
    monkeypatch.setattr(
        trace_scope_mod.global_test_run_manager,
        "update_test_run",
        lambda *_a, **_k: None,
        raising=True,
    )
    monkeypatch.setattr(
        trace_scope_mod.global_test_run_manager,
        "save_test_run",
        lambda *_a, **_k: None,
        raising=True,
    )

    token = current_trace_context.set(trace)
    try:
        result = _assert_test_from_current_trace(
            golden=Golden(input="golden question"),
            metrics=[CapturingMetric("golden question", "trace answer")],
            error_config=ErrorConfig(ignore_errors=False),
            display_config=DisplayConfig(
                show_indicator=False, verbose_mode=False
            ),
        )
    finally:
        current_trace_context.reset(token)

    assert result.input == "golden question"
    assert result.actual_output == "trace answer"
    assert result.metrics_data[0].success is True


def test_assert_test_uses_observe_metrics_for_span_level_evals(monkeypatch):
    app_span = BaseSpan(
        uuid="app",
        status=TraceSpanStatus.SUCCESS,
        children=[],
        trace_uuid="trace",
        parent_uuid="wrapper",
        start_time=perf_counter(),
        end_time=perf_counter(),
        name="retriever",
        input="span question",
        output="span answer",
        metrics=[CapturingMetric("span question", "span answer")],
    )
    trace = _make_pytest_wrapped_trace(app_span)
    captured = {}

    monkeypatch.setattr(
        trace_scope_mod.trace_manager,
        "_convert_span_to_api_span",
        lambda *_: make_span_api_like(),
        raising=True,
    )

    def capture_test_run(api_test_case, *_args, **_kwargs):
        captured["api_test_case"] = api_test_case

    monkeypatch.setattr(
        trace_scope_mod.global_test_run_manager,
        "update_test_run",
        capture_test_run,
        raising=True,
    )
    monkeypatch.setattr(
        trace_scope_mod.global_test_run_manager,
        "save_test_run",
        lambda *_a, **_k: None,
        raising=True,
    )

    token = current_trace_context.set(trace)
    try:
        result = _assert_test_from_current_trace(
            golden=Golden(input="golden question"),
            error_config=ErrorConfig(ignore_errors=False),
            display_config=DisplayConfig(
                show_indicator=False, verbose_mode=False
            ),
        )
    finally:
        current_trace_context.reset(token)

    api_test_case = captured["api_test_case"]
    assert result.success is True
    assert result.metrics_data == []
    assert api_test_case.trace.base_spans[0].metrics_data[0].success is True
