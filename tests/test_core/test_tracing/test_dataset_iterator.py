from tests.test_core.test_tracing.async_app import (
    meta_agent as async_meta_agent,
)
from tests.test_core.test_tracing.sync_app import meta_agent

from deepeval.evaluate.configs import AsyncConfig
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
