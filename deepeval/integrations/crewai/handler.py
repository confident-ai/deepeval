import logging
import deepeval

from typing import Optional, Tuple, Any
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.context import current_span_context, current_trace_context
from deepeval.tracing.tracing import Observer, trace_manager
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


def _get_metrics_data(obj: Any) -> Tuple[Optional[str], Optional[Any]]:
    """Helper to safely extract metrics attached to CrewAI objects."""

    if not obj:
        return None, None
    metric_collection = getattr(obj, "_metric_collection", None)
    metrics = getattr(obj, "_metrics", None)

    if metric_collection is not None or metrics is not None:
        return metric_collection, metrics

    func = getattr(obj, "func", None)
    if func:
        metric_collection = getattr(func, "_metric_collection", None)
        metrics = getattr(func, "_metrics", None)

    return metric_collection, metrics


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

    @staticmethod
    def get_llm_execution_id(source, event) -> str:
        source_id = id(source)
        return f"llm_{source_id}"

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
            output = getattr(
                event, "output", getattr(event, "result", str(event))
            )
            if current_span:
                current_span.output = str(output)
            # set trace output
            current_trace = current_trace_context.get()
            if current_trace:
                current_trace.output = str(output)

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_started(source, event: LLMCallStartedEvent):
            metric_collection, metrics = _get_metrics_data(source)
            observer = Observer(
                span_type="llm",
                func_name="call",
                observe_kwargs={"model": getattr(event, "model", "unknown")},
                metric_collection=metric_collection,
                metrics=metrics,
            )
            self.span_observers[self.get_llm_execution_id(source, event)] = (
                observer
            )
            observer.__enter__()

            if observer.trace_uuid:
                span = trace_manager.get_span_by_uuid(observer.uuid)
                if span:
                    msgs = getattr(event, "messages")
                    span.input = msgs

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_completed(source, event: LLMCallCompletedEvent):
            observer = self.span_observers.pop(
                self.get_llm_execution_id(source, event)
            )
            if observer:
                current_span = current_span_context.get()
                token = None
                span_to_close = trace_manager.get_span_by_uuid(observer.uuid)

                if span_to_close:
                    output = getattr(
                        event, "response", getattr(event, "output", "")
                    )
                    span_to_close.output = output

                    if not current_span or current_span.uuid != observer.uuid:
                        token = current_span_context.set(span_to_close)

                observer.__exit__(None, None, None)

                if token:
                    current_span_context.reset(token)

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
                current_span.output = getattr(
                    event, "output", getattr(event, "result", "")
                )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_started(source, event: ToolUsageStartedEvent):
            metric_collection = None
            metrics = None

            if hasattr(source, "tools"):
                for tools in source.tools:
                    if getattr(tools, "name", None) == event.tool_name:
                        metric_collection, metrics = _get_metrics_data(tools)
                        break

            if not metric_collection:
                agent = getattr(source, "agent", source)
                metric_collection, metrics = _get_metrics_data(agent)

            observer = Observer(
                span_type="tool",
                func_name=event.tool_name,
                function_kwargs=event.tool_args,
                metric_collection=metric_collection,
                metrics=metrics,
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
                token = None
                span_to_close = trace_manager.get_span_by_uuid(observer.uuid)

                if span_to_close:
                    span_to_close.output = getattr(
                        event, "output", getattr(event, "result", None)
                    )

                    if not current_span or current_span.uuid != observer.uuid:
                        token = current_span_context.set(span_to_close)

                observer.__exit__(None, None, None)

                if token:
                    current_span_context.reset(token)

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
                token = None
                span_to_close = trace_manager.get_span_by_uuid(observer.uuid)

                if span_to_close:
                    span_to_close.input = event.query
                    span_to_close.output = event.retrieved_knowledge

                    if not current_span or current_span.uuid != observer.uuid:
                        token = current_span_context.set(span_to_close)

                observer.__exit__(None, None, None)

                if token:
                    current_span_context.reset(token)


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
            wrap_crew_akickoff,
            wrap_crew_akickoff_for_each,
            wrap_agent_execute_task,
            wrap_agent_aexecute_task,
        )

        wrap_crew_kickoff()
        wrap_crew_kickoff_for_each()
        wrap_crew_kickoff_async()
        wrap_crew_kickoff_for_each_async()
        wrap_crew_akickoff()
        wrap_crew_akickoff_for_each()
        wrap_agent_execute_task()
        wrap_agent_aexecute_task()

        IS_WRAPPED_ALL = True
