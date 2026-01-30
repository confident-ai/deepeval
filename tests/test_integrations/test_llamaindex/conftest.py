import pytest
import llama_index.core.instrumentation as instrument
from deepeval.integrations.llama_index import instrument_llama_index

from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.otel.test_exporter import test_exporter
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.tracing.context import current_trace_context, current_span_context


@pytest.fixture(scope="session", autouse=True)
def _setup_llama_index_instrumentation():
    """
    Setup LlamaIndex instrumentation once for all tests in this directory.
    This fixture runs automatically before any tests and only once per test session.
    """
    instrument_llama_index(instrument.get_dispatcher())
    yield


@pytest.fixture(scope="function", autouse=True)
def reset_trace_state():
    trace_manager.clear_traces()
    test_exporter.clear_span_json_list()
    trace_testing_manager.test_dict = None

    current_trace_context.set(None)
    current_span_context.set(None)

    yield
