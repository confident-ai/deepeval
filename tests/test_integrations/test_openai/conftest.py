import pytest


@pytest.fixture(scope="function", autouse=True)
def _setup_openai_instrumentation():
    from deepeval.tracing.otel.setup import (
        ensure_confident_trace,
        shutdown_confident_trace,
    )

    shutdown_confident_trace()
    ensure_confident_trace(("openai",))
    yield
    shutdown_confident_trace()
