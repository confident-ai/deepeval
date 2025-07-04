from typing import Optional
import functools
import deepeval
from deepeval.tracing.attributes import RetrieverAttributes

try:
    from crewai import LLM
    from crewai.utilities.events import (
        CrewKickoffStartedEvent,
        CrewKickoffCompletedEvent,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        ToolUsageFinishedEvent,
    )
    from crewai.utilities.events.base_event_listener import BaseEventListener
    from crewai.memory.memory import Memory

    crewai_installed = True
except:
    crewai_installed = False

from deepeval.tracing import trace_manager
from deepeval.tracing.types import (
    BaseSpan,
    LlmSpan,
    LlmAttributes,
    ToolSpan,
    TraceSpanStatus,
    RetrieverSpan,
)
from uuid import uuid4
from time import perf_counter
from deepeval.telemetry import capture_tracing_integration


def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


class CrewAIEventsListener(BaseEventListener):
    active_trace_id: Optional[str] = None

    def __init__(self):
        capture_tracing_integration("crewai")
        is_crewai_installed()
        super().__init__()

    def setup_listeners(self, crewai_event_bus):

        # patch trace the classes
        self.patch_crewai_LLM("call")
        self.patch_crewai_Memory("search")
        # self.patch_crewai_ToolUsage("use")

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event):
            if self.active_trace_id is None:
                self.active_trace_id = trace_manager.start_new_trace().uuid

            base_span = BaseSpan(
                uuid=str(source.id),
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=self.active_trace_id,
                parent_uuid=None,  # for now crew is the root of the trace
                start_time=perf_counter(),
                name="crew",
                input=event.inputs,
            )
            trace_manager.add_span(base_span)
            trace_manager.add_span_to_trace(base_span)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event):
            base_span = trace_manager.get_span_by_uuid(str(source.id))
            if base_span is None:
                return

            base_span.end_time = perf_counter()
            base_span.status = TraceSpanStatus.SUCCESS
            base_span.output = event.output
            trace_manager.remove_span(base_span.uuid)

            if self.active_trace_id is not None:
                trace_manager.end_trace(self.active_trace_id)
                self.active_trace_id = None

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event: AgentExecutionStartedEvent):
            base_span = BaseSpan(
                uuid=str(event.agent.id),
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=self.active_trace_id,
                parent_uuid=str(event.agent.crew.id),
                start_time=perf_counter(),
                name="(agent) " + event.agent.role,
                metadata={
                    "llm_id": str(
                        id(event.agent.llm)
                    ),  # used to find parent span of llm
                    "agent_key": str(
                        event.agent.key
                    ),  # used to find parent span of tool span
                },
            )

            trace_manager.add_span(base_span)
            trace_manager.add_span_to_trace(base_span)

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event):
            base_span = trace_manager.get_span_by_uuid(str(source.id))
            if base_span is None:
                return

            base_span.end_time = perf_counter()
            base_span.status = TraceSpanStatus.SUCCESS
            trace_manager.remove_span(base_span.uuid)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source, event: ToolUsageFinishedEvent):
            # find the parent span of the tool usage
            target_agent_key = str(event.agent_key)
            matching_span = None
            for span_uuid, span in trace_manager.active_spans.items():
                if (
                    span.metadata
                    and "agent_key" in span.metadata
                    and span.metadata["agent_key"] == target_agent_key
                ):
                    matching_span = span
                    break

            # create a tool span
            tool_span = ToolSpan(
                uuid=str(uuid4()),
                status=TraceSpanStatus.SUCCESS,
                children=[],
                trace_uuid=self.active_trace_id,
                parent_uuid=matching_span.uuid if matching_span else None,
                start_time=event.started_at.timestamp(),  # start time of the tool usage (conver datetime to epoch)
                end_time=event.finished_at.timestamp(),  # end time of the tool usage (conver datetime to epoch)
                name=event.tool_name,  # name of the tool
                input=event.tool_args,  # from the event
                output=event.output,  # from the event
            )

            # add the tool span to the trace
            trace_manager.add_span(tool_span)
            trace_manager.add_span_to_trace(tool_span)

            # remove the tool span from the trace, since it is a completed span
            trace_manager.remove_span(tool_span.uuid)

    def patch_crewai_LLM(self, method_to_patch: str):
        original_methods = {}

        method = getattr(LLM, method_to_patch)
        if callable(method) and not isinstance(method, type):
            original_methods[method_to_patch] = method

            @functools.wraps(method)
            def wrapped_method(*args, original_method=method, **kwargs):
                if self.active_trace_id is None:
                    self.active_trace_id = trace_manager.start_new_trace().uuid

                # find the parent agent if which this LLM instance is a part
                target_llm_id = str(id(args[0]))
                matching_span = None
                for span_uuid, span in trace_manager.active_spans.items():
                    if (
                        span.metadata
                        and "llm_id" in span.metadata
                        and span.metadata["llm_id"] == target_llm_id
                    ):
                        matching_span = span
                        break

                llm_span = LlmSpan(
                    uuid=str(uuid4()),
                    status=TraceSpanStatus.IN_PROGRESS,
                    children=[],
                    trace_uuid=self.active_trace_id,
                    parent_uuid=matching_span.uuid if matching_span else None,
                    start_time=perf_counter(),
                    name="crewai_llm_span_" + str(uuid4()),
                    # TODO: why model is coming unknown?
                    model="unknown",
                    attributes=LlmAttributes(input=args[1], output=""),
                )
                trace_manager.add_span(llm_span)
                trace_manager.add_span_to_trace(llm_span)

                response = original_method(*args, **kwargs)

                llm_span.end_time = perf_counter()
                llm_span.status = TraceSpanStatus.SUCCESS
                llm_span.set_attributes(
                    LlmAttributes(
                        input=llm_span.attributes.input, output=response
                    )
                )
                trace_manager.remove_span(llm_span.uuid)

                return response

            setattr(LLM, method_to_patch, wrapped_method)

    def patch_crewai_Memory(self, method_to_patch: str):
        original_methods = {}
        method = getattr(Memory, method_to_patch)
        if callable(method) and not isinstance(method, type):
            original_methods[method_to_patch] = method

            @functools.wraps(method)
            def wrapped_method(*args, original_method=method, **kwargs):
                # prepare retriver span
                retriever_span = RetrieverSpan(
                    uuid=str(uuid4()),
                    status=TraceSpanStatus.IN_PROGRESS,
                    children=[],
                    trace_uuid=self.active_trace_id,
                    parent_uuid=None,  # none for now, all the memory insances are part of crew,
                    start_time=perf_counter(),
                    name="crewai_retriever_span",
                    embedder="unknown",
                )

                trace_manager.add_span(retriever_span)
                trace_manager.add_span_to_trace(retriever_span)

                response = original_method(*args, **kwargs)
                # end retriever span
                retriever_span.end_time = perf_counter()
                retriever_span.status = TraceSpanStatus.SUCCESS

                # Convert response to List[str] by stringifying each item
                response_str_list = [str(item) for item in response]
                retriever_span.set_attributes(
                    RetrieverAttributes(
                        embedding_input=args[1],
                        retrieval_context=response_str_list,
                    )
                )
                trace_manager.remove_span(retriever_span.uuid)

                return response

            setattr(Memory, method_to_patch, wrapped_method)


def instrumentator(api_key: Optional[str] = None):
    if api_key:
        deepeval.login_with_confident_api_key(api_key)

    CrewAIEventsListener()
