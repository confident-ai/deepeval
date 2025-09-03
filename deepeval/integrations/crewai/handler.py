from typing import Optional
import deepeval
from deepeval.integrations.crewai.agent import (
    Agent as PatchedAgent,
    agent_registry,
)
from deepeval.integrations.crewai.patch import patch_build_context_for_task
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.types import AgentSpan, LlmSpan

try:
    from crewai.crew import Crew
    from crewai.llm import LLM
    from crewai.agent import Agent
    from crewai.utilities.events import AgentExecutionCompletedEvent
    from crewai.utilities.events.base_event_listener import BaseEventListener
    from crewai.task import Task
    from crewai.agents.crew_agent_executor import CrewAgentExecutor
    from crewai.utilities.events import ToolUsageFinishedEvent
    from crewai.tools.tool_usage import ToolUsage
    from crewai.utilities.events import LLMCallCompletedEvent
    from crewai.memory.contextual.contextual_memory import ContextualMemory

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

            if isinstance(current_span, AgentSpan):
                if isinstance(source, Agent):
                    current_span.name = source.role
                    current_span.available_tools = [
                        tool.name for tool in source.tools
                    ]

            if current_span:
                # set llm test case
                input = None
                actual_output = None
                expected_output = None

                if isinstance(event.task, Task):
                    input = event.task.prompt()
                    actual_output = event.output
                    expected_output = event.task.expected_output

                current_span.input = input
                current_span.output = actual_output
                current_span.expected_output = expected_output

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

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source, event: ToolUsageFinishedEvent):
            current_span = current_span_context.get()
            current_span.input = event.tool_args
            current_span.output = event.output
            current_span.name = event.tool_name

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_finished(source, event: LLMCallCompletedEvent):
            current_span = current_span_context.get()

            if isinstance(current_span, LlmSpan):
                if isinstance(source, LLM):
                    current_span.model = source.model

                current_span.input = event.messages
                current_span.output = event.response


def instrument_crewai(api_key: Optional[str] = None):
    is_crewai_installed()
    with capture_tracing_integration("crewai"):
        if api_key:
            deepeval.login(api_key)

        Crew.kickoff = observe(Crew.kickoff)
        LLM.call = observe(LLM.call, type="llm", model="")
        Agent.execute_task = observe(Agent.execute_task, type="agent")
        CrewAgentExecutor.invoke = observe(CrewAgentExecutor.invoke)
        ToolUsage.use = observe(ToolUsage.use, type="tool")
        patch_build_context_for_task()
        CrewAIEventsListener()
