from tests.test_tracing.async_app import meta_agent as async_meta_agent
from tests.test_tracing.sync_app import meta_agent

from deepeval.evaluate import AsyncConfig, evaluate
from deepeval.dataset import Golden

# Define golden inputs
goldens = [
    Golden(input="What's the weather like in SF?"),
    Golden(input="Tell me about Elon Musk."),
]


def test_sync_run_async():
    evaluate(
        goldens=goldens,
        observed_callback=meta_agent,
        async_config=AsyncConfig(run_async=True),
    )
    assert True


def test_sync_run_sync():
    evaluate(
        goldens=goldens,
        observed_callback=meta_agent,
        async_config=AsyncConfig(run_async=False),
    )
    assert True


def test_async_run_async():
    evaluate(
        goldens=goldens,
        observed_callback=async_meta_agent,
        async_config=AsyncConfig(run_async=True),
    )
    assert True


def test_async_run_sync():
    evaluate(
        goldens=goldens,
        observed_callback=async_meta_agent,
        async_config=AsyncConfig(run_async=False),
    )
    assert True
