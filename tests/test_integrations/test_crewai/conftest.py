import pytest
from deepeval.integrations.crewai import instrument_crewai
from deepeval.integrations.crewai.handler import reset_crewai_instrumentation
from deepeval.tracing.tracing import trace_manager

# Import the context variables to reset them
from deepeval.tracing.context import current_trace_context, current_span_context
from deepeval.tracing.otel.test_exporter import test_exporter
from deepeval.tracing.trace_test_manager import trace_testing_manager


@pytest.fixture(scope="session", autouse=True)
def _setup_crewai_instrumentation():
    instrument_crewai()
    yield
    # Add any cleanup code here if needed


@pytest.fixture(autouse=True)
def _clear_traces_between_tests():
    trace_manager.clear_traces()
    test_exporter.clear_span_json_list()
    trace_testing_manager.test_dict = None
    reset_crewai_instrumentation()
    current_trace_context.set(None)
    current_span_context.set(None)

    yield
    trace_manager.clear_traces()
    reset_crewai_instrumentation()
    current_trace_context.set(None)
    current_span_context.set(None)
