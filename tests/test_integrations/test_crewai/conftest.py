import pytest
from deepeval.integrations.crewai import instrument_crewai
from deepeval.tracing import trace_manager


@pytest.fixture(scope="session", autouse=True)
def _setup_crewai_instrumentation():
    instrument_crewai()
    yield
    # Add any cleanup code here if needed


@pytest.fixture(autouse=True)
def _clear_traces_between_tests():
    """Ensure a clean trace state for every test."""
    trace_manager.clear_traces()
    yield