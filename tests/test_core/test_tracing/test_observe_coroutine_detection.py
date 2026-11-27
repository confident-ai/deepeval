import inspect

import pytest

from deepeval.tracing import observe

asgiref_sync = pytest.importorskip("asgiref.sync")


def _blocking(x):
    return x * 2


def test_observe_keeps_sync_to_async_callable_async():
    wrapped = observe()(asgiref_sync.sync_to_async(_blocking))

    assert inspect.iscoroutinefunction(wrapped)


def test_observe_keeps_plain_sync_function_sync():
    wrapped = observe()(_blocking)

    assert not inspect.iscoroutinefunction(wrapped)
