import asyncio
import atexit
import threading
import time

import pytest

import deepeval
from deepeval.tracing import a_flush_traces, flush_traces
from deepeval.tracing.api import TraceApi
from deepeval.tracing.tracing import TraceManager


@pytest.fixture
def make_manager(monkeypatch):
    """Build throwaway TraceManagers and detach them on teardown.

    Each TraceManager registers an atexit warning for traces it never got to
    post, so tests that deliberately leave traces outstanding would spam the
    end of the run with warnings about their own fixtures.
    """
    created = []

    def _make(start_worker: bool = False) -> TraceManager:
        monkeypatch.delenv("CONFIDENT_TRACE_SAMPLE_RATE", raising=False)
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.confident_api_key = "test-key"
        manager._flush_poll_interval = 0.01
        monkeypatch.setattr(
            manager, "_print_trace_status", lambda *a, **k: None, raising=True
        )
        if not start_worker:
            monkeypatch.setattr(
                manager,
                "_ensure_worker_thread_running",
                lambda: None,
                raising=True,
            )
        created.append(manager)
        return manager

    yield _make

    for manager in created:
        atexit.unregister(manager._warn_on_exit)


def make_trace_api(uuid: str = "trace-1") -> TraceApi:
    return TraceApi(
        uuid=uuid,
        startTime="2026-01-01T00:00:00.000000Z",
        endTime="2026-01-01T00:00:01.000000Z",
    )


class TestOutstandingTraces:
    """Traces are counted from enqueue until their send task finishes."""

    def test_starts_at_zero(self, make_manager):
        manager = make_manager()
        assert manager.outstanding_traces == 0

    def test_post_trace_api_increments(self, make_manager):
        manager = make_manager()
        manager.post_trace_api(make_trace_api())
        assert manager.outstanding_traces == 1
        assert manager._trace_queue.qsize() == 1

    def test_post_trace_increments(self, make_manager):
        manager = make_manager()
        trace = manager.start_new_trace(trace_uuid="trace-1")
        manager.post_trace(trace)
        assert manager.outstanding_traces == 1

    def test_sampled_out_trace_is_not_counted(self, make_manager):
        manager = make_manager()
        manager.configure(sampling_rate=0.0)
        manager.post_trace_api(make_trace_api())
        assert manager.outstanding_traces == 0
        assert manager._trace_queue.empty()

    def test_disabled_tracing_is_not_counted(self, make_manager):
        manager = make_manager()
        manager.configure(tracing_enabled=False)
        manager.post_trace_api(make_trace_api())
        assert manager.outstanding_traces == 0

    def test_untrack_never_goes_negative(self, make_manager):
        manager = make_manager()
        manager._untrack_outstanding_trace()
        assert manager.outstanding_traces == 0


class TestFlush:
    """Synchronous flush blocks until nothing is outstanding."""

    def test_returns_true_when_idle(self, make_manager):
        manager = make_manager()
        assert manager.flush(timeout=1.0) is True

    def test_returns_false_when_timeout_expires(self, make_manager):
        manager = make_manager()
        manager.post_trace_api(make_trace_api())
        assert manager.flush(timeout=0.1) is False

    def test_returns_false_immediately_on_zero_timeout(self, make_manager):
        manager = make_manager()
        manager.post_trace_api(make_trace_api())
        assert manager.flush(timeout=0) is False

    def test_waits_until_outstanding_trace_clears(self, make_manager):
        manager = make_manager()
        manager.post_trace_api(make_trace_api())

        def clear_later():
            time.sleep(0.1)
            manager._untrack_outstanding_trace()

        threading.Thread(target=clear_later, daemon=True).start()

        started = time.perf_counter()
        assert manager.flush(timeout=5.0) is True
        assert time.perf_counter() - started >= 0.1

    def test_does_not_return_during_queue_handoff(self, make_manager):
        """The window between dequeue and task registration still blocks."""
        manager = make_manager()
        manager.post_trace_api(make_trace_api())
        manager._trace_queue.get_nowait()

        assert manager._trace_queue.empty()
        assert not manager._in_flight_tasks
        assert manager.flush(timeout=0.1) is False


class TestAFlush:
    """Async flush yields to the loop instead of blocking it."""

    async def test_returns_true_when_idle(self, make_manager):
        manager = make_manager()
        assert await manager.a_flush(timeout=1.0) is True

    async def test_returns_false_when_timeout_expires(self, make_manager):
        manager = make_manager()
        manager.post_trace_api(make_trace_api())
        assert await manager.a_flush(timeout=0.1) is False

    async def test_waits_until_outstanding_trace_clears(self, make_manager):
        manager = make_manager()
        manager.post_trace_api(make_trace_api())

        async def clear_later():
            await asyncio.sleep(0.1)
            manager._untrack_outstanding_trace()

        asyncio.get_running_loop().create_task(clear_later())

        assert await manager.a_flush(timeout=5.0) is True

    async def test_lets_other_coroutines_run_while_waiting(self, make_manager):
        manager = make_manager()
        manager.post_trace_api(make_trace_api())
        ticks = []

        async def ticker():
            for _ in range(3):
                await asyncio.sleep(0.01)
                ticks.append(1)

        task = asyncio.get_running_loop().create_task(ticker())
        assert await manager.a_flush(timeout=0.2) is False
        await task
        assert len(ticks) == 3


class TestWorkerDrain:
    """End to end: the worker thread clears the count once traces are sent."""

    async def test_flush_returns_after_worker_posts_trace(
        self, make_manager, monkeypatch
    ):
        sent_bodies = []

        class RecordingApi:
            def __init__(self, api_key=None):
                self.api_key = api_key

            async def a_send_request(self, method, endpoint, body):
                sent_bodies.append(body)
                return None, "https://app.confident-ai.com/trace/1"

        monkeypatch.setattr(
            "deepeval.tracing.tracing.Api", RecordingApi, raising=True
        )

        manager = make_manager(start_worker=True)
        manager.post_trace_api(make_trace_api())

        assert manager.outstanding_traces == 1
        assert await manager.a_flush(timeout=10.0) is True
        assert manager.outstanding_traces == 0
        assert len(sent_bodies) == 1


class TestPublicApi:
    """flush_traces / a_flush_traces delegate to the global trace manager."""

    def test_exported_from_deepeval(self):
        assert deepeval.flush_traces is flush_traces
        assert deepeval.a_flush_traces is a_flush_traces

    def test_flush_traces_forwards_timeout(self, monkeypatch):
        recorded = {}

        def fake_flush(timeout):
            recorded["timeout"] = timeout
            return True

        monkeypatch.setattr(
            "deepeval.tracing.tracing.trace_manager.flush",
            fake_flush,
            raising=True,
        )
        assert flush_traces(timeout=12.5) is True
        assert recorded["timeout"] == 12.5

    async def test_a_flush_traces_forwards_timeout(self, monkeypatch):
        recorded = {}

        async def fake_a_flush(timeout):
            recorded["timeout"] = timeout
            return False

        monkeypatch.setattr(
            "deepeval.tracing.tracing.trace_manager.a_flush",
            fake_a_flush,
            raising=True,
        )
        assert await a_flush_traces(timeout=7.5) is False
        assert recorded["timeout"] == 7.5

    def test_default_timeout_is_thirty_seconds(self):
        import inspect

        assert (
            inspect.signature(flush_traces).parameters["timeout"].default
            == 30.0
        )
        assert (
            inspect.signature(a_flush_traces).parameters["timeout"].default
            == 30.0
        )
