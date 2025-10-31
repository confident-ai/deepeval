import asyncio
import importlib
import pytest
import os

from deepeval.evaluate.configs import CacheConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.execute import _a_execute_agentic_test_case
from deepeval.test_run.test_run import TestRun, TestRunManager
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.llms.openai_model import GPTModel
from deepeval.dataset.golden import Golden


exec_mod = importlib.import_module("deepeval.evaluate.execute")


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="needs OPENAI_API_KEY",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("ignore_errors", [True, False])
async def test_agentic_async_persists_metric_on_cancel(
    monkeypatch, ignore_errors
):
    """
    Even if the agentic eval coroutine is cancelled (e.g., by an outer wait_for),
    _a_execute_agentic_test_case should still persist a TestCase entry in the TestRun.
    """

    # build a metric and patch its a_measure to hang
    metric = AnswerRelevancyMetric(model=GPTModel(model="gpt-5"))

    async def sleepy_a_measure(*args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(metric, "a_measure", sleepy_a_measure, raising=True)

    trm = TestRunManager()
    tr = TestRun(identifier="persist-on-cancel-agentic")
    trm.set_test_run(tr)

    # Golden with an observed callback that hangs
    golden = Golden(input="ping")

    async def sleepy_observed_callback(_text: str):
        await asyncio.sleep(10)

    # run the agentic case and cancel fast to simulate outer timeout.
    coroutine = asyncio.wait_for(
        _a_execute_agentic_test_case(
            golden=golden,
            test_run_manager=trm,
            test_results=[],
            count=0,
            verbose_mode=None,
            ignore_errors=ignore_errors,
            skip_on_missing_params=False,
            show_indicator=False,  # avoid Rich progress noise in CI
            _use_bar_indicator=False,
            _is_assert_test=False,
            observed_callback=sleepy_observed_callback,
            trace=None,
            trace_metrics=[metric],  # give it a trace level metric
            progress=None,
            pbar_id=None,
        ),
        timeout=0.05,
    )
    if ignore_errors:
        await coroutine
    else:
        with pytest.raises(asyncio.TimeoutError):
            await coroutine

    # assert that we persisted the case into the TestRun.
    recorded = trm.get_test_run()
    assert recorded is not None
    assert len(recorded.test_cases) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestCollectionWarning")
def test_agentic_sync_persists_on_timeout_ignore_errors_true(
    monkeypatch, settings
):
    """Sync agentic path: when ignore_errors=True, a timeout should not raise,
    but the test case must still be persisted."""
    # configure a quick timeout window
    with settings.edit(persist=False):
        # ensure we don't rely on SDK retries and we keep the runner simple
        settings.DEEPEVAL_SDK_RETRY_PROVIDERS = []
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1
        # cut the outer budget so the observed callback hits the timeout
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 0.05

    # minimal TestRun plumbing (avoid disk)
    trm = TestRunManager()
    trm.save_to_disk = False  # prevents the need for hidden .deepeval dir and avoids disk writes
    tr = TestRun(identifier="persist-on-timeout-agentic-sync")
    trm.set_test_run(tr)

    # patch in our own TestRunManager so we can inspect persisted results
    monkeypatch.setattr(exec_mod, "global_test_run_manager", trm, raising=True)

    # agentic callback that hangs longer than our outer deadline
    async def sleepy_observed_callback(_text: str):
        await asyncio.sleep(3600)

    # run the agentic sync flow with ignore_errors=True, this should not raise
    exec_mod.execute_agentic_test_cases(
        goldens=[Golden(input="ping")],
        observed_callback=sleepy_observed_callback,
        display_config=DisplayConfig(
            show_indicator=False
        ),  # avoid Rich progress noise in CI
        cache_config=CacheConfig(
            write_cache=False
        ),  # avoid reading or writing to our hidden dir
        error_config=ErrorConfig(ignore_errors=True),
        identifier="persist-on-timeout-agentic-sync",
        _use_bar_indicator=False,
    )

    # assert the case was persisted despite the timeout
    recorded = trm.get_test_run()
    assert recorded is not None
    assert len(recorded.test_cases) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestCollectionWarning")
def test_agentic_sync_persists_on_timeout_ignore_errors_false(
    monkeypatch, settings
):
    """Sync agentic path: when ignore_errors=False, we should raise TimeoutError
    after persisting the test case."""
    # configure a quick timeout window
    with settings.edit(persist=False):
        # ensure we don't rely on SDK retries and we keep the runner simple
        settings.DEEPEVAL_SDK_RETRY_PROVIDERS = []
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 1
        # cut the outer budget so the observed callback hits the timeout
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 0.05

    trm = TestRunManager()
    trm.save_to_disk = False  # prevents the need for hidden .deepeval dir and avoids disk writes
    tr = TestRun(identifier="persist-on-timeout-agentic-sync-raise")
    trm.set_test_run(tr)

    # patch in our own TestRunManager so we can inspect persisted results
    monkeypatch.setattr(exec_mod, "global_test_run_manager", trm, raising=True)

    # agentic callback that hangs longer than our outer deadline
    async def sleepy_observed_callback(_text: str):
        await asyncio.sleep(3600)

    # with ignore_errors=False, the runner should raise after marking/persisting
    with pytest.raises(asyncio.TimeoutError):
        exec_mod.execute_agentic_test_cases(
            goldens=[Golden(input="ping")],
            observed_callback=sleepy_observed_callback,
            display_config=DisplayConfig(show_indicator=False),
            cache_config=CacheConfig(write_cache=False),
            error_config=ErrorConfig(ignore_errors=False),
            identifier="persist-on-timeout-agentic-sync-raise",
            _use_bar_indicator=False,
        )

    # assert the case was persisted despite the timeout
    recorded = trm.get_test_run()
    assert recorded is not None
    assert len(recorded.test_cases) == 1
