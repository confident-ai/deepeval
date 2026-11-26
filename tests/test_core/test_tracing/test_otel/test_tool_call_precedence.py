"""An explicitly empty OTel tool list survives local span collection."""

import pytest

from deepeval.tracing.otel.bridge import DeepEvalSpanBridge
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import EvalMode, EvalSession


@pytest.fixture
def collecting_tracer(provider, monkeypatch):
    session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    monkeypatch.setattr(trace_manager, "eval_session", session)
    provider.add_span_processor(DeepEvalSpanBridge())
    try:
        yield provider.get_tracer("test"), session.traces_to_evaluate
    finally:
        provider.shutdown()
        trace_manager.clear_traces()


@pytest.mark.parametrize(
    "declared, expected",
    [
        (None, ["lookup"]),
        ([], []),
        ("[]", []),
        (['{"name":"explicit"}'], ["explicit"]),
        (["malformed"], ["lookup"]),
    ],
    ids=["absent", "native-empty", "json-empty", "nonempty", "invalid"],
)
def test_agent_tool_list_precedes_child_tool_fallback(
    collecting_tracer, declared, expected
):
    tracer, traces = collecting_tracer
    attrs = {"gen_ai.operation.name": "invoke_agent"}
    if declared is not None:
        attrs["confident.span.tools_called"] = declared

    with tracer.start_as_current_span("planner", attributes=attrs):
        with tracer.start_as_current_span(
            "lookup",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "lookup",
            },
        ):
            pass

    (trace,) = traces
    (agent,) = trace.root_spans
    assert [call.name for call in agent.tools_called] == expected
    assert [call.name for call in trace.tools_called] == ["lookup"]


@pytest.mark.parametrize(
    "declared, expected",
    [
        (None, ["lookup"]),
        ([], []),
        (['{"name":"explicit"}'], ["explicit"]),
        (["malformed"], ["lookup"]),
    ],
    ids=["absent", "explicit-empty", "nonempty", "invalid"],
)
def test_openinference_tool_list_precedes_adapter_fallback(
    collecting_tracer, declared, expected
):
    tracer, traces = collecting_tracer
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "lookup",
    }
    if declared is not None:
        attrs["confident.span.tools_called"] = declared

    with tracer.start_as_current_span("model", attributes=attrs):
        pass

    (trace,) = traces
    (span,) = trace.root_spans
    assert span.integration == "OpenInference"
    assert [call.name for call in span.tools_called] == expected
    api = trace_manager.create_trace_api(trace).model_dump(by_alias=True)
    assert [
        call["name"] for call in api["llmSpans"][0]["toolsCalled"]
    ] == expected
