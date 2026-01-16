import pytest
from deepeval.tracing.trace_test_manager import trace_testing_manager


@pytest.fixture(autouse=True)
def enable_trace_testing_mode(request):
    prev_name = getattr(trace_testing_manager, "test_name", None)
    prev_dict = getattr(trace_testing_manager, "test_dict", None)

    # Always start clean
    trace_testing_manager.test_dict = None

    # If decorators didn't set a test_name, set one so non-snapshot tests capture payloads too.
    if not trace_testing_manager.test_name:
        trace_testing_manager.test_name = request.node.nodeid

    yield

    trace_testing_manager.test_name = prev_name
    trace_testing_manager.test_dict = prev_dict


@pytest.fixture(autouse=True)
def mock_openai_chat_completion(monkeypatch, request):
    """
    Patch the module global `client` in the test module currently executing.
    """
    test_mod = request.module

    # If a given test module doesn't define `client`, do nothing.
    if not hasattr(test_mod, "client"):
        return

    class _Msg:
        content = "MOCK_RESPONSE"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    def _create(*args, **kwargs):
        return _Resp()

    # Patch the method on the exact client instance referenced by your_llm_app()
    monkeypatch.setattr(
        test_mod.client.chat.completions, "create", _create, raising=True
    )


# More edge cases
# agent specific spans to be tested
# cover all span types, tools, retrievers
# combination of metrics and no metrics on these different spans
#
