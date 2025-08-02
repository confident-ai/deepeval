from tests.test_core.test_tracing.async_app import (
    meta_agent as async_meta_agent,
)
from tests.test_core.test_tracing.sync_app import meta_agent

from deepeval.evaluate import AsyncConfig, dataset, test_run
from deepeval.dataset import Golden
import asyncio


# Define golden inputs
goldens = [
    Golden(input="What's the weather like in SF?"),
    Golden(input="Tell me about Elon Musk."),
]


def test_async_run_async():
    for golden in dataset(
        goldens=goldens, async_config=AsyncConfig(run_async=True)
    ):
        task = asyncio.create_task(async_meta_agent(golden.input))
        test_run.append(task)
    assert True


def test_sync_run_async():
    for golden in dataset(
        goldens=goldens, async_config=AsyncConfig(run_async=True)
    ):
        meta_agent(golden.input)
    assert True


def test_sync_run_sync():
    for golden in dataset(
        goldens=goldens, async_config=AsyncConfig(run_async=False)
    ):
        meta_agent(golden.input)
    assert True


# TODO: fix the bug here
# def test_async_run_sync():
#     for golden in dataset(
#         goldens=goldens, async_config=AsyncConfig(run_async=False)
#     ):
#         task = asyncio.create_task(async_meta_agent(golden.input))
#         test_run.append(task)
#     assert True
