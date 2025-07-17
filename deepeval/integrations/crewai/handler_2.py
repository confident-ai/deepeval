from typing import Optional
import uuid
import deepeval
from deepeval.tracing import trace_manager
from deepeval.tracing.types import (
    BaseSpan,
    TraceSpanStatus,
)
from time import perf_counter

try:
    from crewai.utilities.events import (
        CrewKickoffStartedEvent,
        CrewKickoffCompletedEvent,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
    )
    from crewai.utilities.events.base_event_listener import BaseEventListener
    from crewai.crew import Crew
    crewai_installed = True
except:
    crewai_installed = False

def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


class CrewAIEventsListener(BaseEventListener):
    active_trace_id: Optional[str] = None

    def __init__(self):
        deepeval.capture_tracing_integration("crewai")
        is_crewai_installed()
        super().__init__()
    
    def end_trace(self, base_span: BaseSpan):
        current_trace = trace_manager.get_trace_by_uuid(self.active_trace_id)
        if current_trace is not None:
            current_trace.input = base_span.input
            current_trace.output = base_span.output
        trace_manager.end_trace(self.active_trace_id)
        self.active_trace_id = None

    def end_span(self, base_span: BaseSpan):
        base_span.end_time = perf_counter()
        base_span.status = TraceSpanStatus.SUCCESS
        trace_manager.remove_span(str(base_span.uuid))

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source: Crew, event: CrewKickoffStartedEvent):
            if self.active_trace_id is None:
                self.active_trace_id = trace_manager.start_new_trace().uuid
            
            base_span = BaseSpan(
                uuid=str(uuid.uuid4()),
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=self.active_trace_id,
                parent_uuid=None,  # for now crew is the root of the trace
                start_time=perf_counter(),
                name="Crew",
                input=event.inputs,
                metadata={
                    "Crew.id": source.id,
                }
            )
            trace_manager.add_span(base_span)
            trace_manager.add_span_to_trace(base_span)
        
        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source: Crew, event: CrewKickoffCompletedEvent):
            if self.active_trace_id is None:
                return
            
            base_span = None
            for span_uuid, span in trace_manager.active_spans.items():
                if span.metadata.get("Crew.id") == source.id:
                    base_span = span
                    break
            
            if base_span is None:
                return
            base_span.output = event.output
            self.end_span(base_span)
            self.end_trace(base_span)

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event):
            print(">>agent started")

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event):
            print(">>agent completed")

def instrumentator(api_key: Optional[str] = None):
    if api_key:
        deepeval.login_with_confident_api_key(api_key)

    CrewAIEventsListener()