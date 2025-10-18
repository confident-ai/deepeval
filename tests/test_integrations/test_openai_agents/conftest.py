import pytest
from agents import add_trace_processor
from deepeval.openai_agents.callback_handler import DeepEvalTracingProcessor


@pytest.fixture(scope="session", autouse=True)
def _install_deepeval_tracer():
    # guard in case something else already registered it
    try:
        from openai.agents.provider import trace_processors

        if any(
            isinstance(tp, DeepEvalTracingProcessor) for tp in trace_processors
        ):
            yield
            return
    except Exception:
        pass

    proc = DeepEvalTracingProcessor()
    add_trace_processor(proc)
    yield
    # if the SDK exposes a remove API, you could remove here
