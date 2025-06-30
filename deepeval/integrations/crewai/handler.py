from typing import Optional
import functools

try:
    from crewai import LLM
    from crewai.tools.tool_usage import ToolUsage
    from crewai.utilities.events import (
        CrewKickoffStartedEvent,
        CrewKickoffCompletedEvent,
        LLMCallStartedEvent,
        LLMCallCompletedEvent,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        AgentExecutionErrorEvent,
        TaskStartedEvent,
        TaskCompletedEvent
    )
    from crewai.utilities.events.base_event_listener import BaseEventListener
    crewai_installed = True
except:
    crewai_installed = False

from deepeval.tracing import trace_manager
from deepeval.tracing.types import BaseSpan, LlmSpan, LlmAttributes, ToolSpan, TraceSpanStatus
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
        # self.patch_crewai_LLM("call")
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
                parent_uuid=None, # for now crew is the root of the trace
                start_time=perf_counter(),
                name = "crew",
                input=event.inputs
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
        def on_agent_started(source, event):
            base_span = BaseSpan(
                uuid=str(source.id),
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=self.active_trace_id,
                parent_uuid=str(source.crew.id),
                start_time=perf_counter(),
                name="(agent) "+source.role
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

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source, event):
            base_span = BaseSpan(
                uuid=str(source.__hash__),
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=self.active_trace_id,
                parent_uuid=str(source.agent.id),
                start_time=perf_counter(),
                name="(task) "+ source.description
            )
            trace_manager.add_span(base_span)
            trace_manager.add_span_to_trace(base_span)

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event):
            base_span = trace_manager.get_span_by_uuid(str(source.__hash__))
            if base_span is None:
                return
            
            base_span.end_time = perf_counter()
            base_span.status = TraceSpanStatus.SUCCESS
            trace_manager.remove_span(base_span.uuid)

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_started(source, event):
            pass
            # print("--------------llm started------------------")
            # print(source)
            # print("--------------------------------")
            # print(event)
            # print("--------------------------------")
        
        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_completed(source, event):
            pass
            # print("--------------llm completed------------------")
            # print(source)
            # print("--------------------------------")
            # print(event)
            # print("--------------------------------")
        
    def patch_crewai_LLM(self, method_to_patch: str):    
        original_methods = {}

        method = getattr(LLM, method_to_patch)
        if callable(method) and not isinstance(method, type):
            original_methods[method_to_patch] = method

            @functools.wraps(method)
            def wrapped_method(*args, original_method=method, **kwargs):                
                if self.active_trace_id is None:
                    self.active_trace_id = trace_manager.start_new_trace().uuid
                
                llm_span = LlmSpan(
                    uuid=str(uuid4()),
                    status=TraceSpanStatus.IN_PROGRESS,
                    children=[],
                    trace_uuid=self.active_trace_id,
                    parent_uuid=None,
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
                    LlmAttributes(input=llm_span.attributes.input, output=response)
                )
                trace_manager.remove_span(llm_span.uuid)
                
                return response
            
            setattr(LLM, method_to_patch, wrapped_method)

    def patch_crewai_ToolUsage(self, method_to_patch: str):

        original_methods = {}

        method = getattr(ToolUsage, method_to_patch)
        if callable(method) and not isinstance(method, type):
            original_methods[method_to_patch] = method

            @functools.wraps(method)
            def wrapped_method(*args, original_method=method, **kwargs):
                tool_calling = args[1]
                
                tool_span = ToolSpan(
                    uuid=str(uuid4()),
                    status=TraceSpanStatus.IN_PROGRESS,
                    children=[],
                    trace_uuid=self.active_trace_id,
                    parent_uuid=None,
                    start_time=perf_counter(),
                    name=tool_calling.tool_name,
                    input=tool_calling.arguments,
                )
                trace_manager.add_span(tool_span)
                trace_manager.add_span_to_trace(tool_span)
                
                response = original_method(*args, **kwargs)
                
                tool_span.end_time = perf_counter()
                tool_span.status = TraceSpanStatus.SUCCESS
                tool_span.output = response

                trace_manager.remove_span(tool_span.uuid)
                
                return response

            setattr(ToolUsage, method_to_patch, wrapped_method)