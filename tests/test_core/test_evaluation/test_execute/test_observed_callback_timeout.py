import asyncio
import pytest

from deepeval.evaluate.execute import (
    execute_agentic_test_cases,
    a_execute_agentic_test_cases,
)
from deepeval.dataset import Golden


def test_observed_callback_times_out_sync_path(monkeypatch, settings):
    """
    Ensures async observed_callback in the sync path is bounded by
    DEEPEVAL_PER_TASK_TIMEOUT_SECONDS and raises asyncio.TimeoutError.
    """
    with settings.edit(persist=False):
        # Make the timeout tiny so the test is fast
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 1
        settings.DEEPEVAL_LOG_STACK_TRACES = True

    async def slow_callback(_):
        # Sleep well past the configured timeout
        await asyncio.sleep(5)

    goldens = [Golden(input="hello")]

    with pytest.raises(asyncio.TimeoutError):
        execute_agentic_test_cases(
            goldens=goldens, observed_callback=slow_callback
        )


@pytest.mark.asyncio
async def test_observed_callback_times_out_async_path(monkeypatch, settings):
    """
    Ensures async observed_callback in the async path is bounded by
    DEEPEVAL_PER_TASK_TIMEOUT_SECONDS and raises asyncio.TimeoutError.
    """

    with settings.edit(persist=False):
        # Make the timeout tiny so the test is fast
        settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = 1

    async def slow_callback(_):
        # Sleep well past the configured timeout
        await asyncio.sleep(5)

    goldens = [Golden(input="hello")]

    with pytest.raises(asyncio.TimeoutError):
        await a_execute_agentic_test_cases(
            goldens=goldens,
            observed_callback=slow_callback,
        )


def test_observed_callback_sync_callback_unaffected():
    """
    Sanity check: synchronous callbacks still run fine and do not raise.
    """

    def sync_callback(_):
        return "ok"

    goldens = [Golden(input="hello")]
    # should not raise
    execute_agentic_test_cases(goldens=goldens, observed_callback=sync_callback)
