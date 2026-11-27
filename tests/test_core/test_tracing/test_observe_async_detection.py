import asyncio
import pytest

from deepeval.tracing import observe


@pytest.mark.asyncio
async def test_observe_async_function_avoids_asyncio_iscoroutinefunction(
    monkeypatch,
):
    def fail_if_called(_obj):
        raise AssertionError("asyncio.iscoroutinefunction should not be called")

    monkeypatch.setattr(asyncio, "iscoroutinefunction", fail_if_called)

    @observe()
    async def sample(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    assert await sample(1) == 2
