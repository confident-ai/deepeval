import asyncio
import pytest


from tests.test_core.test_tracing.async_app import (
    meta_agent as async_meta_agent,
)
from tests.test_core.test_tracing.sync_app import meta_agent

from deepeval.evaluate.configs import (
    AsyncConfig,
    DisplayConfig,
    CacheConfig,
    ErrorConfig,
)
from deepeval.evaluate import execute as exec_mod
from deepeval.dataset import EvaluationDataset, Golden


# Define golden inputs
goldens = [
    Golden(input="What's the weather like in SF?"),
    Golden(input="Tell me about Elon Musk."),
]


def test_async_run_async():
    dataset = EvaluationDataset(goldens=goldens)
    for golden in dataset.evals_iterator(
        async_config=AsyncConfig(run_async=True)
    ):
        dataset.evaluate(async_meta_agent(golden.input))
    assert True


def test_sync_run_async():
    dataset = EvaluationDataset(goldens=goldens)
    for golden in dataset.evals_iterator(
        async_config=AsyncConfig(run_async=True)
    ):
        meta_agent(golden.input)
    assert True


def test_sync_run_sync():
    dataset = EvaluationDataset(goldens=goldens)
    for golden in dataset.evals_iterator(
        async_config=AsyncConfig(run_async=False)
    ):
        meta_agent(golden.input)
    assert True


def test_no_leftovers_runs_trace_eval(monkeypatch):
    called = {"trace_eval": False}

    async def _fake_a_evaluate_traces(*a, **k):
        called["trace_eval"] = True

    async def _fake_snapshot_tasks():
        return set()

    monkeypatch.setattr(
        exec_mod, "_a_evaluate_traces", _fake_a_evaluate_traces, raising=False
    )
    monkeypatch.setattr(
        exec_mod, "_snapshot_tasks", _fake_snapshot_tasks, raising=False
    )

    # build the iterator that uses evaluate_test_cases
    ds = EvaluationDataset(goldens=[Golden(input="x")])
    gen = ds.evals_iterator(
        async_config=AsyncConfig(run_async=True),
        display_config=DisplayConfig(show_indicator=False, verbose_mode=False),
        cache_config=CacheConfig(write_cache=False),
        error_config=ErrorConfig(
            ignore_errors=False, skip_on_missing_params=False
        ),
    )

    # executor yields the first golden and patches asyncio.create_task
    next(gen)

    # ensure execute.py sees a pending trace to evaluate
    exec_mod.trace_manager.traces_to_evaluate.clear()
    exec_mod.trace_manager.traces_to_evaluate.append(object())

    # schedule one trivial task so we enter create_task
    async def _noop():
        await asyncio.sleep(0)

    async def _schedule_one():
        asyncio.create_task(_noop())
        await asyncio.sleep(0)

    asyncio.get_event_loop().run_until_complete(_schedule_one())

    # finish iterator which should run _a_evaluate_traces
    with pytest.raises(StopIteration):
        next(gen)

    assert (
        called["trace_eval"] is True
    ), "trace eval skipped when there were no leftovers"


def test_snapshot_tasks_runtimeerror_still_runs_trace_eval(monkeypatch):
    """
    _snapshot_tasks() raises RuntimeError in the finally block.
    We should still evaluate traces.
    """
    called = {"trace_eval": False}

    async def _fake_a_evaluate_traces(*a, **k):
        called["trace_eval"] = True

    # first call we will let the snapshot succeed
    # on the second call we will raise a RuntimeError
    # this happens in the `evaluate_test_cases` finally block, right before evaluating traces
    calls = {"n": 0}

    async def _flaky_snapshot_tasks():
        calls["n"] += 1
        if calls["n"] == 1:
            return set()
        raise RuntimeError("loop is closing")

    monkeypatch.setattr(
        exec_mod, "_a_evaluate_traces", _fake_a_evaluate_traces, raising=False
    )
    monkeypatch.setattr(
        exec_mod, "_snapshot_tasks", _flaky_snapshot_tasks, raising=False
    )

    ds = EvaluationDataset(goldens=[Golden(input="x")])
    gen = ds.evals_iterator(
        async_config=AsyncConfig(run_async=True),
        display_config=DisplayConfig(show_indicator=False, verbose_mode=False),
        cache_config=CacheConfig(write_cache=False),
        error_config=ErrorConfig(
            ignore_errors=False, skip_on_missing_params=False
        ),
    )

    # executor yields the first golden and patches asyncio.create_task
    next(gen)

    # ensure traces are pending for evaluation
    exec_mod.trace_manager.traces_to_evaluate.clear()
    exec_mod.trace_manager.traces_to_evaluate.append(object())

    # schedule one trivial task so we enter create_task
    async def _noop():
        await asyncio.sleep(0)

    async def _schedule_one():
        asyncio.create_task(_noop())
        await asyncio.sleep(0)

    asyncio.get_event_loop().run_until_complete(_schedule_one())

    # in finally phase flaky snapshot triggers RuntimeError on second call
    # but we should still run _a_evaluate_traces when this happens
    with pytest.raises(StopIteration):
        next(gen)

    assert (
        called["trace_eval"] is True
    ), "trace eval skipped after RuntimeError from _snapshot_tasks()"


def test_closed_loop_skips_trace_eval(monkeypatch):
    """
    Force the loop to report closed in the executor's finally, so trace
    evaluation is skipped. We can't do trace evaluation if the loop has
    closed for some reason.
    """
    called = {"trace_eval": False}

    async def _fake_a_evaluate_traces(*a, **k):
        called["trace_eval"] = True  # should not run

    async def _safe_snapshot_tasks():
        return set()

    monkeypatch.setattr(
        exec_mod, "_a_evaluate_traces", _fake_a_evaluate_traces, raising=False
    )
    monkeypatch.setattr(
        exec_mod, "_snapshot_tasks", _safe_snapshot_tasks, raising=False
    )

    ds = EvaluationDataset(goldens=[Golden(input="x")])
    gen = ds.evals_iterator(
        async_config=AsyncConfig(run_async=True),
        display_config=DisplayConfig(show_indicator=False, verbose_mode=False),
        cache_config=CacheConfig(write_cache=False),
        error_config=ErrorConfig(
            ignore_errors=False, skip_on_missing_params=False
        ),
    )

    # executor yields the first golden and patches asyncio.create_task
    next(gen)

    # make sure there will be at least one created task so we hit the finally block
    async def _noop():
        await asyncio.sleep(0)

    async def _schedule_one():
        asyncio.create_task(_noop())
        await asyncio.sleep(0)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_schedule_one())

    exec_mod.trace_manager.traces_to_evaluate.clear()
    exec_mod.trace_manager.traces_to_evaluate.append(object())

    # now force the loop to appear closed for the finally guard
    import asyncio.base_events as be

    monkeypatch.setattr(
        be.BaseEventLoop, "is_closed", lambda self: True, raising=False
    )

    with pytest.raises(StopIteration):
        next(gen)

    assert (
        called["trace_eval"] is False
    ), "trace eval should NOT run when event loop is closed"
