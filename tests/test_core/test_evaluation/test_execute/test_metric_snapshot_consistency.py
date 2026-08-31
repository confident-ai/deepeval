import asyncio

import pytest

from deepeval.evaluate.execute import _a_execute_llm_test_cases
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run.test_run import TestRun, TestRunManager


class SlowMetric(BaseMetric):
    def __init__(self):
        self.threshold = 0.5
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
        return "SlowMetric"

    async def a_measure(self, test_case, *args, **kwargs):
        await asyncio.sleep(0.2)
        self.score = 1.0
        self.reason = "slow-done"
        self.success = True

    def is_successful(self):
        return bool(self.success)


class BoomMetric(BaseMetric):
    def __init__(self):
        self.threshold = 0.5
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
        return "BoomMetric"

    async def a_measure(self, test_case, *args, **kwargs):
        raise RuntimeError("boom")

    def is_successful(self):
        return bool(self.success)


@pytest.mark.asyncio
async def test_async_metric_snapshot_waits_for_siblings():
    slow = SlowMetric()
    boom = BoomMetric()

    trm = TestRunManager()
    tr = TestRun(identifier="snapshot-consistency")
    trm.set_test_run(tr)

    with pytest.raises(RuntimeError, match="boom"):
        await _a_execute_llm_test_cases(
            metrics=[slow, boom],
            test_case=LLMTestCase(input="x", actual_output="y"),
            test_run_manager=trm,
            test_results=[],
            count=0,
            test_run=tr,
            ignore_errors=False,
            skip_on_missing_params=False,
            use_cache=False,
            show_indicator=False,
            _use_bar_indicator=False,
            _is_assert_test=False,
            progress=None,
            pbar_id=None,
        )

    recorded = trm.get_test_run()
    assert recorded is not None
    assert len(recorded.test_cases) == 1

    metric_data_by_name = {
        metric_data.name: metric_data
        for metric_data in recorded.test_cases[0].metrics_data or []
    }

    slow_snapshot = metric_data_by_name["SlowMetric"]
    assert slow_snapshot.score == 1.0
    assert slow_snapshot.reason == "slow-done"
    assert slow_snapshot.success is True

    await asyncio.sleep(0.3)

    assert slow.score == 1.0
    assert slow.reason == "slow-done"
    assert slow.success is True
