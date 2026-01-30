import logging
import deepeval
from collections import defaultdict
from time import perf_counter
from typing import Optional, Tuple, Any, List, Union
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.context import current_span_context, current_trace_context
from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.tracing.types import ToolSpan, SpanType, TraceSpanStatus
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

# GLOBAL STATE to prevent duplicate listeners
IS_WRAPPED_ALL = False
_listener_instance = None


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
        self.tool_observers_stack: dict[str, List[Union[Observer, None]]] = (
            defaultdict(list)
        )

    def reset_state(self):
        """Clears all internal state to prevent pollution between tests."""
        self.span_observers.clear()
        self.tool_observers_stack.clear()

    @staticmethod
    def get_tool_stack_key(source, tool_name) -> str:
        """
        Generates a unique key for the tool stack.
        FIX: Uses role/name instead of id() to be robust against object copying by CrewAI.
        """
        identifier = getattr(
            source, "role", getattr(source, "name", str(id(source)))
        )
        return f"{tool_name}_{identifier}"

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

    def _flatten_tool_span(self, span):
        """
        Callback to move any child ToolSpans up to the parent.
        """
        if not span.parent_uuid or not span.children:
            return

        parent_span = trace_manager.get_span_by_uuid(span.parent_uuid)
        if not parent_span:
            return

        # Identify child tool spans (ghost nesting)
        tools_to_move = [
            child for child in span.children if isinstance(child, ToolSpan)
        ]

        if tools_to_move:
            if parent_span.children is None:
                parent_span.children = []

            for child in tools_to_move:
                child.parent_uuid = parent_span.uuid
                parent_span.children.append(child)

            span.children = [
                child
                for child in span.children
                if not isinstance(child, ToolSpan)
            ]

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            current_span = current_span_context.get()
            if current_span:
                current_span.input = event.inputs
            current_trace = current_trace_context.get()
            if current_trace:
                current_trace.input = event.inputs

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            current_span = current_span_context.get()
            output = getattr(
                event, "output", getattr(event, "result", str(event))
            )
            if current_span:
                current_span.output = str(output)
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
            key = self.get_llm_execution_id(source, event)
            if key in self.span_observers:
                observer = self.span_observers.pop(key)
                if observer:
                    current_span = current_span_context.get()
                    token = None
                    span_to_close = trace_manager.get_span_by_uuid(
                        observer.uuid
                    )

                    if span_to_close:
                        output = getattr(
                            event, "response", getattr(event, "output", "")
                        )
                        span_to_close.output = output
                        if (
                            not current_span
                            or current_span.uuid != observer.uuid
                        ):
                            token = current_span_context.set(span_to_close)

                    observer.__exit__(None, None, None)
                    if token:
                        current_span_context.reset(token)

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event: AgentExecutionStartedEvent):
            current_span = current_span_context.get()
            if current_span:
                current_span.input = event.task_prompt

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event: AgentExecutionCompletedEvent):
            current_span = current_span_context.get()
            if current_span:
                current_span.output = getattr(
                    event, "output", getattr(event, "result", "")
                )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_started(source, event: ToolUsageStartedEvent):
            key = self.get_tool_stack_key(source, event.tool_name)

            # 1. Internal Stack Check
            if self.tool_observers_stack[key]:
                self.tool_observers_stack[key].append(None)
                return

            # 2. SMART DEDUPING
            current_span = current_span_context.get()
            span_type = getattr(current_span, "type", None)
            is_tool_span = span_type == "tool" or span_type == SpanType.TOOL
            if (
                is_tool_span
                and getattr(current_span, "name", "") == event.tool_name
            ):
                self.tool_observers_stack[key].append(None)
                return

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

            self.tool_observers_stack[key].append(observer)
            observer.__enter__()

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_completed(source, event: ToolUsageFinishedEvent):
            key = self.get_tool_stack_key(source, event.tool_name)
            observer = None

            if (
                key in self.tool_observers_stack
                and self.tool_observers_stack[key]
            ):
                item = self.tool_observers_stack[key].pop()
                if item is None:
                    return
                observer = item

            if not observer:
                current_span = current_span_context.get()
                if (
                    current_span
                    and getattr(current_span, "type", None)
                    in ["tool", SpanType.TOOL]
                    and getattr(current_span, "name", "") == event.tool_name
                ):
                    current_span.output = getattr(
                        event, "output", getattr(event, "result", None)
                    )

                    if current_span.end_time is None:
                        current_span.end_time = perf_counter()

                    current_span.status = TraceSpanStatus.SUCCESS

                    self._flatten_tool_span(current_span)
                    trace_manager.remove_span(current_span.uuid)

                    if current_span.parent_uuid:
                        parent = trace_manager.get_span_by_uuid(
                            current_span.parent_uuid
                        )
                        current_span_context.set(parent if parent else None)
                    else:
                        current_span_context.set(None)
                    return

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

                observer.update_span_properties = self._flatten_tool_span
                observer.__exit__(None, None, None)

                if span_to_close and span_to_close.end_time is None:
                    span_to_close.end_time = perf_counter()
                    span_to_close.status = TraceSpanStatus.SUCCESS

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
            key = self.get_knowledge_execution_id(source, event)
            if key in self.span_observers:
                observer = self.span_observers.pop(key)
                if observer:
                    current_span = current_span_context.get()
                    token = None
                    span_to_close = trace_manager.get_span_by_uuid(
                        observer.uuid
                    )

                    if span_to_close:
                        span_to_close.input = event.query
                        span_to_close.output = event.retrieved_knowledge

                        if (
                            not current_span
                            or current_span.uuid != observer.uuid
                        ):
                            token = current_span_context.set(span_to_close)

                    observer.__exit__(None, None, None)

                    if token:
                        current_span_context.reset(token)


def instrument_crewai(api_key: Optional[str] = None):
    global _listener_instance

    is_crewai_installed()
    with capture_tracing_integration("crewai"):
        if api_key:
            deepeval.login(api_key)

        wrap_all()

        if _listener_instance is None:
            _listener_instance = CrewAIEventsListener()


def reset_crewai_instrumentation():
    global _listener_instance
    if _listener_instance:
        _listener_instance.reset_state()


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
