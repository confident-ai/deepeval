from typing import Optional
import deepeval
from deepeval.integrations.crewai.agent import (
    Agent as PatchedAgent,
    agent_registry,
)

try:
    from crewai.crew import Crew
    from crewai.llm import LLM
    from crewai.agent import Agent
    from crewai.utilities.events import AgentExecutionCompletedEvent
    from crewai.utilities.events.base_event_listener import BaseEventListener
    from crewai.task import Task

    crewai_installed = True
except:
    crewai_installed = False


def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.tracing.tracing import (
    observe,
    current_span_context,
    trace_manager,
    current_trace_context,
)


class CrewAIEventsListener(BaseEventListener):
    def __init__(self):
        is_crewai_installed()
        super().__init__()

    def setup_listeners(self, crewai_event_bus):

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(
            source, event: AgentExecutionCompletedEvent
        ):
            current_span = current_span_context.get()
            if current_span:
                # set llm test case
                input = None
                output = None
                expected_output = None

                if isinstance(event.task, Task):
                    input = event.task.prompt()
                    actual_output = event.output
                    expected_output = event.task.expected_output

                current_span.llm_test_case = LLMTestCase(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                )

                # set metrics
                if isinstance(source, PatchedAgent):
                    current_span.metrics = agent_registry.get_metrics(source)
                    current_span.metric_collection = (
                        agent_registry.get_metric_collection(source)
                    )

                    # set offline evals
                    if current_span.metric_collection:
                        trace_manager.integration_traces_to_evaluate.append(
                            current_trace_context.get()
                        )


def instrumentator(api_key: Optional[str] = None):
    is_crewai_installed()
    if api_key:
        deepeval.login(api_key)

    Crew.kickoff = observe(Crew.kickoff)
    LLM.call = observe(LLM.call)
    Agent.execute_task = observe(Agent.execute_task)
    CrewAIEventsListener()
