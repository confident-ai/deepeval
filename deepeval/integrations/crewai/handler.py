import deepeval
from typing import Optional, cast
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.context import current_span_context
from deepeval.tracing.tracing import Observer
from deepeval.tracing.types import LlmSpan

try:
    from crewai.utilities.events.base_event_listener import BaseEventListener
    from crewai.events import CrewKickoffStartedEvent, CrewKickoffCompletedEvent, LLMCallStartedEvent, LLMCallCompletedEvent
    crewai_installed = True
except:
    crewai_installed = False


def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )

class CrewAIEventsListener(BaseEventListener):
    def __init__(self):
        is_crewai_installed()
        super().__init__()
        self.span_observers: dict[str, Observer] = {}

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            # Assuming that this event is called in the crew.kickoff method
            current_span = current_span_context.get()
            
            # set the input
            if current_span:
                current_span.input = event.inputs

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            # Assuming that this event is called in the crew.kickoff method
            current_span = current_span_context.get()
            # set the output
            if current_span:
                current_span.output = event.output

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_started(source, event: LLMCallStartedEvent):
            # Assuming that this event is called in the llm.call method
            current_span = current_span_context.get()
            
            # set the input
            if current_span:
                current_span.input = event.messages

                # set the model
                if isinstance(current_span, LlmSpan):
                    current_span.model = event.model
                
        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_completed(source, event: LLMCallCompletedEvent):
            # Assuming that this event is called in the llm.call method
            current_span = current_span_context.get()
            
            # set the output
            if current_span:
                current_span.output = event.response


def instrument_crewai(api_key: Optional[str] = None):
    is_crewai_installed()
    with capture_tracing_integration("crewai"):
        if api_key:
            deepeval.login(api_key)

        from deepeval.integrations.crewai.wrapper import wrap_crew_kickoff, wrap_llm_call
        wrap_crew_kickoff()
        wrap_llm_call()
        # Crew.kickoff = observe(Crew.kickoff)
        # LLM.call = observe(LLM.call, type="llm", model="")
        # Agent.execute_task = observe(Agent.execute_task, type="agent")
        # CrewAgentExecutor.invoke = observe(CrewAgentExecutor.invoke)
        # ToolUsage.use = observe(ToolUsage.use, type="tool")
        # patch_build_context_for_task()
        CrewAIEventsListener()
