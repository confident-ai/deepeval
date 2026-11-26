import asyncio
import gc

import pytest

from deepeval.dataset import Golden
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.execute import loop as loop_module


def _run_executor_with_loop_closed_during_cleanup():
    """Drive the async agentic executor so its task-cleanup ``finally`` block
    runs while the event loop is already closed and an exception is
    propagating out of the ``try``.

    A task is scheduled (so the ``if created_tasks:`` cleanup block is
    entered), then the loop is closed. When the executor reaches the gather
    step it raises ``RuntimeError("Event loop is closed")``, which must
    propagate out of the ``finally``. The previous ``return`` inside that
    ``finally`` swallowed it, so the generator completed silently instead.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        it = loop_module.a_execute_agentic_test_cases_from_loop(
            goldens=[Golden(input="hi")],
            trace_metrics=None,
            test_results=[],
            loop=loop,
            display_config=DisplayConfig(show_indicator=False),
            async_config=AsyncConfig(run_async=True),
            error_config=ErrorConfig(
                ignore_errors=True, skip_on_missing_params=True
            ),
        )

        # First step runs the baseline snapshot and yields the golden.
        next(it)

        async def _noop():
            return None

        # ``asyncio.create_task`` is monkeypatched by the executor to register
        # the task, so ``created_tasks`` becomes non-empty. Let it finish so
        # nothing is left pending when the loop is closed.
        task = asyncio.create_task(_noop())
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))

        loop.close()

        # Draining the generator now reaches the cleanup path on a closed loop.
        for _ in it:
            pass
    finally:
        asyncio.set_event_loop(None)
        if not loop.is_closed():
            loop.close()


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited")
def test_exception_propagates_when_loop_closed_during_cleanup():
    with pytest.raises(RuntimeError) as exc_info:
        _run_executor_with_loop_closed_during_cleanup()
    assert "closed" in str(exc_info.value).lower()

    # The raised traceback pins the never-awaited wait_for() coroutine from the
    # gather step; collect it now (while this test's warning filter is active)
    # so it doesn't surface as a stray RuntimeWarning during later teardown.
    del exc_info
    gc.collect()
