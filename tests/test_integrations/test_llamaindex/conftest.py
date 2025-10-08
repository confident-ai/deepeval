import pytest
import llama_index.core.instrumentation as instrument
from deepeval.integrations.llama_index import instrument_llama_index


@pytest.fixture(scope="session", autouse=True)
def _setup_llama_index_instrumentation():
    """
    Setup LlamaIndex instrumentation once for all tests in this directory.
    This fixture runs automatically before any tests and only once per test session.
    """
    instrument_llama_index(instrument.get_dispatcher())
    yield
    # Add any cleanup code here if needed
