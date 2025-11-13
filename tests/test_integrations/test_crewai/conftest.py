import importlib
import pytest

# Skip the whole package if CrewAI or its dependencies fail import
try:
    importlib.import_module("crewai")
    importlib.import_module("chromadb")  # may fail due to sqlite requirement
except Exception as e:
    pytest.skip(
        f"Skipping CrewAI integration tests: {e}", allow_module_level=True
    )

from deepeval.integrations.crewai import instrument_crewai


@pytest.fixture(scope="session", autouse=True)
def _setup_crewai_instrumentation():
    instrument_crewai()
    yield
    # Add any cleanup code here if needed
