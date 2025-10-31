import asyncio
import importlib
import time
import os
import pytest

from deepeval.evaluate.evaluate import evaluate as run_evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.evaluate.execute import (
    _a_execute_llm_test_cases,
)
from deepeval.test_case import LLMTestCase
from deepeval.test_run.test_run import TestRun, TestRunManager
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.llms.openai_model import GPTModel


exec_mod = importlib.import_module("deepeval.evaluate.execute")
pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("ignore_errors", [True, False])
async def test_llm_async_persists_metric_on_cancel(monkeypatch, ignore_errors):
    """
    Even if the test-case coroutine is cancelled (e.g., by a gather/outer timeout),
    _a_execute_llm_test_cases must still persist MetricData and update the TestRun.
    """

    # build a normal metric instance, then patch its a_measure to hang
    metric = AnswerRelevancyMetric(model=GPTModel(model="gpt-5"))

    async def sleepy_a_measure(*args, **kwargs):
        # simulate a provider call that takes too long
        await asyncio.sleep(3600)

    monkeypatch.setattr(metric, "a_measure", sleepy_a_measure, raising=True)

    trm = TestRunManager()
    tr = TestRun(identifier="persist-on-cancel")
    trm.set_test_run(tr)

    test_case = LLMTestCase(input="ping", actual_output="pong")
    metrics = [metric]

    # run the LLM async case but cut it off quickly
    # the test results should still be recorded and the test case should be counted
    coroutine = asyncio.wait_for(
        _a_execute_llm_test_cases(
            metrics=metrics,
            test_case=test_case,
            test_run_manager=trm,
            test_results=[],
            count=0,
            test_run=tr,
            ignore_errors=ignore_errors,
            skip_on_missing_params=False,
            use_cache=False,
            show_indicator=False,  # avoid Rich progress noise in CI
            _use_bar_indicator=False,
            _is_assert_test=False,
            progress=None,
            pbar_id=None,
        ),
        timeout=0.05,  # small timeout
    )
    if ignore_errors:
        await coroutine
    else:
        with pytest.raises(asyncio.TimeoutError):
            await coroutine

    # assert the test run has one case with one metric recorded as errored
    recorded = trm.get_test_run()
    assert recorded is not None
    assert len(recorded.test_cases) == 1

    tc = recorded.test_cases[0]
    assert tc.metrics_data is not None and len(tc.metrics_data) == 1

    md = tc.metrics_data[0]
    # error and success=False should be set by safe_a_measure
    assert md.error
    assert md.success is False


@pytest.mark.filterwarnings("ignore::pytest.PytestCollectionWarning")
def test_llm_sync_persists_metric_on_timeout_ignore_errors_true(
    monkeypatch, settings
):
    """Sync LLM path: when ignore_errors=True, a timeout should not raise,
    but the test case must still be persisted with the metric marked as errored.
    """
    # configure a quick timeout window
    with settings.edit(persist=False):
        # ensure we don't rely on SDK retries and we keep the runner simple
        settings.DEEPEVAL_SDK_RETRY_PROVIDERS = []
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1
        # cut the outer budget so run_sync_with_timeout triggers
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 0.05

    # Metric whose sync path blocks
    metric = AnswerRelevancyMetric(model=GPTModel(model="gpt-5"))

    def sleepy_measure(*args, **kwargs):
        # simulate a stuck provider call
        # this timeout must be marger thatn our configured DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE
        time.sleep(10)

    # patch sync path
    monkeypatch.setattr(metric, "measure", sleepy_measure, raising=True)

    trm = TestRunManager()
    trm.save_to_disk = False  # prevents the need for hidden .deepeval dir and avoids disk writes
    tr = TestRun(identifier="persist-on-timeout-sync")
    trm.set_test_run(tr)

    # patch in our own TestRunManager so we can inspect persisted results
    monkeypatch.setattr(
        exec_mod,
        "global_test_run_manager",
        trm,
        raising=True,
    )

    # build the test case and run the sync flow
    case = LLMTestCase(input="ping", actual_output="pong")

    # run_async=False ensures we go down sync codepath
    # cache_config=CacheConfig(write_cache=False) required to avoid reading from hidden dir
    run_evaluate(
        [case],
        metrics=[metric],
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=True),
        cache_config=CacheConfig(write_cache=False),
    )

    # assert the case was persisted despite the timeout
    recorded = trm.get_test_run()
    assert recorded is not None
    assert len(recorded.test_cases) == 1

    tc = recorded.test_cases[0]
    assert tc.metrics_data is not None and len(tc.metrics_data) == 1

    md = tc.metrics_data[0]
    assert (md.success is False and md.error) or (md.success is None)


@pytest.mark.filterwarnings("ignore::pytest.PytestCollectionWarning")
def test_llm_sync_persists_metric_on_timeout_ignore_errors_false(
    monkeypatch, settings
):
    """Sync LLM path: when ignore_errors=False we should raise TimeoutError
    after marking the metric; the test case must still be persisted."""
    # configure a quick timeout window
    with settings.edit(persist=False):
        # ensure we don't rely on SDK retries and we keep the runner simple
        settings.DEEPEVAL_SDK_RETRY_PROVIDERS = []
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1
        # cut the outer budget so run_sync_with_timeout triggers
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 0.05

    # metric whose sync path blocks
    metric = AnswerRelevancyMetric(model=GPTModel(model="gpt-5"))

    def sleepy_measure(*args, **kwargs):
        # simulate a stuck provider call
        # this timeout must be marger thatn our configured DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE
        time.sleep(10)

    # patch sync path
    monkeypatch.setattr(metric, "measure", sleepy_measure, raising=True)

    trm = TestRunManager()
    trm.save_to_disk = False  # prevents the need for hidden .deepeval dir and avoids disk writes
    tr = TestRun(identifier="persist-on-timeout-sync")
    trm.set_test_run(tr)

    # patch in our own TestRunManager so we can inspect persisted results
    monkeypatch.setattr(
        exec_mod,
        "global_test_run_manager",
        trm,
        raising=True,
    )

    # build the test case and run the sync flow
    case = LLMTestCase(input="ping", actual_output="pong")

    with pytest.raises(asyncio.TimeoutError):
        # run_async=False ensures we go down sync codepath
        # cache_config=CacheConfig(write_cache=False) required to avoid reading from hidden dir
        run_evaluate(
            [case],
            metrics=[metric],
            async_config=AsyncConfig(run_async=False),
            error_config=ErrorConfig(ignore_errors=False),
            cache_config=CacheConfig(write_cache=False),
        )

    # assert the case was persisted despite the timeout
    recorded = trm.get_test_run()
    assert recorded is not None
    assert len(recorded.test_cases) == 1

    tc = recorded.test_cases[0]
    assert tc.metrics_data is not None and len(tc.metrics_data) == 1

    md = tc.metrics_data[0]
    assert (md.success is False and md.error) or (md.success is None)
