"""wrap_up_test_run must not divide by zero when no test case has a verdict."""

from types import SimpleNamespace

from deepeval.test_run.test_run import TestRunManager


def _verdictless_case(name: str = "t") -> SimpleNamespace:
    return SimpleNamespace(
        success=None,
        metrics=[],
        metrics_data={},
        metric_data=None,
        name=name,
        trace_id=None,
        span_id=None,
        component_id=None,
        span_start_timestamp=None,
        trace=None,
        order=0,
    )


def test_wrap_up_survives_all_none_success():
    """A run where every metric is verdict-less (no thresholds) reaches the
    summary with test_passed == test_failed == 0 — the code explicitly keeps
    going in that state (see the "All metrics errored" warning above it), so
    the Pass Rate line must degrade to N/A instead of ZeroDivisionError."""
    mgr = TestRunManager()
    mgr.create_test_run()
    mgr.test_run.test_cases = [_verdictless_case()]
    mgr.test_run.conversational_test_cases = []
    mgr.save_to_disk = False
    mgr.save_test_run = lambda *a, **k: None

    mgr.wrap_up_test_run(0.5)  # must not raise
