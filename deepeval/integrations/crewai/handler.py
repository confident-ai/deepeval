import deepeval
from typing import Optional
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.tracing import Observer

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
            print(f"Crew '{event.crew_name}' has started execution!")

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            print(f"Crew '{event.crew_name}' has completed execution!")


def instrument_crewai(api_key: Optional[str] = None):
    is_crewai_installed()
    with capture_tracing_integration("crewai"):
        if api_key:
            deepeval.login(api_key)

        Crew.kickoff = observe(Crew.kickoff)
        LLM.call = observe(LLM.call, type="llm", model="")
        # Agent.execute_task = observe(Agent.execute_task, type="agent")
        # CrewAgentExecutor.invoke = observe(CrewAgentExecutor.invoke)
        # ToolUsage.use = observe(ToolUsage.use, type="tool")
        # patch_build_context_for_task()
        CrewAIEventsListener()
