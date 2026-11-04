from crewai.events import (
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    ToolUsageErrorEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)

import deepeval.integrations.crewai.handler as crewai_handler
from deepeval.integrations.crewai.handler import CrewAIEventsListener
from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.tracing.types import TraceSpanStatus


class FakeLLM:
    model = "gpt-4o-mini"


class FakeAgent:
    role = "Weather Reporter"


def _emit(source, event):
    # The event bus dispatches listeners through a thread pool, so join the
    # returned future before asserting on listener state.
    future = crewai_event_bus.emit(source, event)
    if future is not None:
        future.result()


def test_llm_call_failed_event_closes_llm_span_and_finalizes_trace():
    listener = crewai_handler._listener_instance
    source = FakeLLM()

    with Observer(span_type="agent", func_name="crew-kickoff") as root:
        trace_uuid = root.trace_uuid
        started_event = LLMCallStartedEvent(
            call_id="call-1", model="gpt-4o-mini", messages=[]
        )
        _emit(source, started_event)

        # The LLM span is open alongside the root span.
        assert len(trace_manager.active_spans) == 2
        key = CrewAIEventsListener.get_llm_execution_id(source, started_event)
        observer = listener.span_observers[key]
        llm_span = trace_manager.get_span_by_uuid(observer.uuid)

        # CrewAI emits LLMCallFailedEvent instead of LLMCallCompletedEvent
        # when the call fails, then re-raises the error.
        _emit(
            source,
            LLMCallFailedEvent(
                call_id="call-1", model="gpt-4o-mini", error="rate limited"
            ),
        )

    # The LLM span must not leak, and the trace must finalize once the root
    # span exits.
    assert len(trace_manager.active_spans) == 0
    assert trace_uuid not in trace_manager.active_traces
    assert listener.span_observers == {}

    # The failed call is recorded as errored, not left open.
    assert llm_span.end_time is not None
    assert llm_span.status == TraceSpanStatus.ERRORED
    assert llm_span.error == "rate limited"


def test_tool_usage_error_event_clears_pending_tool_metadata():
    listener = crewai_handler._listener_instance
    source = FakeAgent()

    _emit(
        source,
        ToolUsageStartedEvent(
            tool_name="get_weather", tool_args={"city": "London"}
        ),
    )
    assert any(listener.tool_observers_stack.values())

    # CrewAI emits ToolUsageErrorEvent instead of ToolUsageFinishedEvent
    # when the tool raises.
    _emit(
        source,
        ToolUsageErrorEvent(
            tool_name="get_weather",
            tool_args={"city": "London"},
            error="tool crashed",
        ),
    )

    assert not any(listener.tool_observers_stack.values())
