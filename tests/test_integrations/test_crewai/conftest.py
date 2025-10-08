import pytest
from deepeval.integrations.crewai import instrument_crewai


@pytest.fixture(scope="session", autouse=True)
def _setup_crewai_instrumentation():
    instrument_crewai()
    yield
    # Add any cleanup code here if needed
