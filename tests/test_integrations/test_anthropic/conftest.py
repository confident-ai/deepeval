import pytest

from tests.test_integrations.utils import confident_trace_instrumentation


@pytest.fixture(scope="function", autouse=True)
def _setup_anthropic_instrumentation():
    with confident_trace_instrumentation("anthropic"):
        yield
