import logging
import deepeval

from typing import Optional
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.context import current_span_context, current_trace_context
from deepeval.tracing.tracing import Observer
from deepeval.tracing.types import LlmSpan
from deepeval.config.settings import get_settings


logger = logging.getLogger(__name__)


try:
    from crewai.events import BaseEventListener
    from crewai.events import (
        CrewKickoffStartedEvent,
        CrewKickoffCompletedEvent,
        LLMCallStartedEvent,
        LLMCallCompletedEvent,
        AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
        KnowledgeRetrievalStartedEvent,
        KnowledgeRetrievalCompletedEvent,
    )

    crewai_installed = True
except ImportError as e:
    if get_settings().DEEPEVAL_VERBOSE_MODE:
        if isinstance(e, ModuleNotFoundError):
            logger.warning(
                "Optional crewai dependency not installed: %s",
                e.name,
                stacklevel=2,
            )
        else:
            logger.warning(
                "Optional crewai import failed: %s",
                e,
                stacklevel=2,
            )

    crewai_installed = False

IS_WRAPPED_ALL = False


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

    @staticmethod
    def get_tool_execution_id(source, event) -> str:
        source_id = id(source)
        task_id = getattr(event, "task_id", "unknown")
        agent_id = getattr(event, "agent_id", "unknown")
        tool_name = getattr(event, "tool_name", "unknown")
        execution_id = f"tool_{source_id}_{task_id}_{agent_id}_{tool_name}"

        return execution_id

    @staticmethod
    def get_knowledge_execution_id(source, event) -> str:
        source_id = id(source)
        agent_id = id(event.agent) if hasattr(event, "agent") else "unknown"
        execution_id = f"_knowledge_{source_id}_{agent_id}"

        return execution_id

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            # Assuming that this event is called in the crew.kickoff method
            current_span = current_span_context.get()

            # set the input
            if current_span:
                current_span.input = event.inputs

            # set trace input
            current_trace = current_trace_context.get()
            if current_trace:
                current_trace.input = event.inputs

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            # Assuming that this event is called in the crew.kickoff method
            current_span = current_span_context.get()

            # set the output
            if current_span:
                current_span.output = str(event.output)

            # set trace output
            current_trace = current_trace_context.get()
            if current_trace:
                current_trace.output = str(event.output)

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

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event: AgentExecutionStartedEvent):
            # Assuming that this event is called in the agent.execute_task method
            current_span = current_span_context.get()

            # set the input
            if current_span:
                current_span.input = event.task_prompt

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event: AgentExecutionCompletedEvent):
            # Assuming that this event is called in the agent.execute_task method
            current_span = current_span_context.get()

            # set the output
            if current_span:
                current_span.output = event.output

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_started(source, event: ToolUsageStartedEvent):
            observer = Observer(
                span_type="tool",
                func_name=event.tool_name,
                function_kwargs=event.tool_args,
            )
            self.span_observers[self.get_tool_execution_id(source, event)] = (
                observer
            )
            observer.__enter__()

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_completed(source, event: ToolUsageFinishedEvent):
            observer = self.span_observers.pop(
                self.get_tool_execution_id(source, event)
            )
            if observer:
                current_span = current_span_context.get()
                if current_span:
                    current_span.output = event.output
                observer.__exit__(None, None, None)

        @crewai_event_bus.on(KnowledgeRetrievalStartedEvent)
        def on_knowledge_started(source, event: KnowledgeRetrievalStartedEvent):
            observer = Observer(
                span_type="tool",
                func_name="knowledge_retrieval",
                function_kwargs={},
            )
            self.span_observers[
                self.get_knowledge_execution_id(source, event)
            ] = observer
            observer.__enter__()

        @crewai_event_bus.on(KnowledgeRetrievalCompletedEvent)
        def on_knowledge_completed(
            source, event: KnowledgeRetrievalCompletedEvent
        ):
            observer = self.span_observers.pop(
                self.get_knowledge_execution_id(source, event)
            )
            if observer:
                current_span = current_span_context.get()
                if current_span:
                    current_span.input = event.query
                    current_span.output = event.retrieved_knowledge
                observer.__exit__(None, None, None)


def instrument_crewai(api_key: Optional[str] = None):
    is_crewai_installed()
    with capture_tracing_integration("crewai"):
        if api_key:
            deepeval.login(api_key)

        wrap_all()

        CrewAIEventsListener()


def wrap_all():
    global IS_WRAPPED_ALL

    if not IS_WRAPPED_ALL:
        from deepeval.integrations.crewai.wrapper import (
            wrap_crew_kickoff,
            wrap_crew_kickoff_for_each,
            wrap_crew_kickoff_async,
            wrap_crew_kickoff_for_each_async,
            wrap_llm_call,
            wrap_agent_execute_task,
        )

        wrap_crew_kickoff()
        wrap_crew_kickoff_for_each()
        wrap_crew_kickoff_async()
        wrap_crew_kickoff_for_each_async()
        wrap_llm_call()
        wrap_agent_execute_task()

        IS_WRAPPED_ALL = True
