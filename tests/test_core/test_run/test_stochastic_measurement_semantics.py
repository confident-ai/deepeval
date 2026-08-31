"""Offline regression for #3110: stochastic judge measurement semantics.

Each ``TestRun`` is one measurement; re-runs are new measurements with no
cross-run holder. ``flaky`` is a user declaration, not an observed instability
signal, and cached runs are replays that must be bypassed for genuine
re-measurement. Tests below are deterministic and require no LLM calls.
"""

import pytest
import statistics

from deepeval.evaluate.configs import CacheConfig
from deepeval.test_run.api import LLMApiTestCase
from deepeval.test_run.test_run import TestRun
from deepeval.tracing.api import MetricData


def make_metric_data(success, score=None, flaky: bool = False) -> MetricData:
    return MetricData(name="GEval", success=success, score=score, flaky=flaky)


def make_api_case(name: str = "case") -> LLMApiTestCase:
    return LLMApiTestCase(name=name, input="hello", order=0)


def test_one_run_is_one_measurement_no_cross_run_holder():
    """Two independent runs of the same case have isolated verdicts.

    There is no field on TestRun or LLMApiTestCase that points at another
    run — each run's ``success`` is computed only from its own metric_data
    via the AND-fold in ``update_metric_data``.
    """
    c1 = make_api_case()
    c1.update_metric_data(make_metric_data(True, score=0.62))
    c2 = make_api_case()
    c2.update_metric_data(make_metric_data(False, score=0.58))

    assert c1.success is True
    assert c2.success is False
    # No cross-run attribute exists
    assert not hasattr(c1, "previous_run")
    assert not hasattr(c2, "previous_run")
    # TestRun itself has no cross-run pointer either
    tr = TestRun(identifier="run-1")
    assert not hasattr(tr, "previous_run")
    assert not hasattr(tr, "linked_runs")


def test_stochastic_scores_straddle_threshold_produce_distribution_not_canonical_verdict():
    """Scores 0.62/0.58/0.61 against threshold 0.6 yield a distribution.

    The useful artifact is pass-rate + spread, not a single canonical boolean.
    """
    threshold = 0.6
    scores = [0.62, 0.58, 0.61]
    successes = [s >= threshold for s in scores]
    assert successes == [True, False, True]
    pass_rate = sum(successes) / len(successes)
    assert pass_rate == pytest.approx(2 / 3)
    mean = statistics.mean(scores)
    stdev = statistics.pstdev(scores)
    assert mean == pytest.approx(0.603333, rel=1e-3)
    assert stdev > 0
    # threshold inside observed spread flags a flickering case
    assert min(scores) < threshold < max(scores)
    # Three-state summary
    if all(successes):
        state = "always-passed"
    elif not any(successes):
        state = "always-failed"
    else:
        state = "mixed"
    assert state == "mixed"


def test_threshold_inside_spread_flag():
    """Helper logic for the future opt-in n-runs reporting."""
    def threshold_inside_spread(scores, threshold):
        return min(scores) < threshold < max(scores)

    assert threshold_inside_spread([0.62, 0.58, 0.61], 0.6) is True
    assert threshold_inside_spread([0.9, 0.92, 0.88], 0.6) is False
    assert threshold_inside_spread([0.1, 0.2, 0.15], 0.6) is False
    # Degenerate: single measurement has no spread
    assert threshold_inside_spread([0.62], 0.6) is False


def test_flaky_is_declaration_not_observed_instability():
    """``flaky=True`` never decides pass/fail, regardless of success value."""
    c = make_api_case()
    c.update_metric_data(make_metric_data(False, flaky=True))
    assert c.success is None
    # A later non-flaky pass still decides
    c.update_metric_data(make_metric_data(True, flaky=False))
    assert c.success is True
    # But even a failing flaky does not override the prior pass
    c.update_metric_data(make_metric_data(False, flaky=True))
    assert c.success is True
    # MetricData is still retained for auditability
    assert len(c.metrics_data) == 3
    assert c.metrics_data[0].flaky is True


def test_cache_bypass_required_for_true_re_measurement():
    """Defaults ensure re-runs are real measurements, not cache replays."""
    cfg = CacheConfig()
    assert cfg.use_cache is False
    assert cfg.write_cache is True
    # Docstring must clarify the replay vs re-measurement distinction
    assert "replays" in CacheConfig.__doc__ or "replay" in CacheConfig.__doc__.lower()
    assert "stochastic" in CacheConfig.__doc__.lower() or "measurement" in CacheConfig.__doc__.lower()


def test_case_verdict_is_and_fold_not_canonical_merge():
    """Within one run, success folds by AND; across runs there is no fold."""
    c = make_api_case()
    c.update_metric_data(make_metric_data(True, score=0.62))
    c.update_metric_data(make_metric_data(True, score=0.61))
    assert c.success is True
    c.update_metric_data(make_metric_data(False, score=0.58))
    assert c.success is False
    # Further True cannot revive it — AND semantics
    c.update_metric_data(make_metric_data(True, score=0.9))
    assert c.success is False
