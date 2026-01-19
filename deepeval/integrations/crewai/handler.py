import logging
import deepeval

from contextlib import contextmanager
from typing import Optional, Dict, Any
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.context import current_span_context, current_trace_context
from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.tracing.types import LlmSpan, BaseSpan, Trace
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
    """
    Event listener for CrewAI that captures tracing spans for tools and knowledge retrieval.

    IMPORTANT: CrewAI events may fire from different execution contexts (threads/tasks)
    than where the Observer was created. ContextVars don't propagate across these
    boundaries, so we must:
    1. Capture the trace/span context when creating an Observer
    2. Restore that context before calling __exit__ on the Observer
    3. Use direct span lookups via trace_manager instead of relying on ContextVars
    """

    def __init__(self):
        is_crewai_installed()
        super().__init__()
        # Store both Observer and captured context for each span
        # Key: execution_id, Value: dict with 'observer', 'trace', 'parent_span'
        self.span_data: Dict[str, Dict[str, Any]] = {}

    @contextmanager
    def _restore_context(
        self,
        stored_data: Optional[Dict[str, Any]] = None,
        span: Optional[BaseSpan] = None,
    ):
        """
        Context manager to restore trace and span context for event callbacks
        that may execute in different contexts than where the Observer was created.

        Similar to LangChain's _ctx() pattern in callback.py

        Args:
            stored_data: Dict containing 'observer', 'trace', 'parent_span' for Observer-based spans
            span: Direct span reference to restore (for non-Observer event handlers)
        """
        trace_token = None
        span_token = None

        try:
            # Determine which trace and span to restore
            if stored_data:
                stored_trace = stored_data.get("trace")
                observer = stored_data.get("observer")
                target_span = (
                    trace_manager.get_span_by_uuid(observer.uuid)
                    if observer and observer.uuid
                    else None
                )
            else:
                stored_trace = None
                target_span = span

            # Restore trace context if needed
            if stored_trace:
                current_trace = current_trace_context.get()
                if (
                    current_trace is None
                    or current_trace.uuid != stored_trace.uuid
                ):
                    # Verify trace is still active
                    if stored_trace.uuid in trace_manager.active_traces:
                        trace_token = current_trace_context.set(stored_trace)

            # Restore span context
            # This is critical: __exit__ expects current_span_context to hold the span being exited
            if target_span:
                current_span = current_span_context.get()
                if (
                    current_span is None
                    or current_span.uuid != target_span.uuid
                ):
                    span_token = current_span_context.set(target_span)

            yield

        finally:
            if span_token is not None:
                current_span_context.reset(span_token)
            if trace_token is not None:
                current_trace_context.reset(trace_token)

    def _capture_current_context(self) -> Dict[str, Any]:
        """Capture the current trace and span context to restore later."""
        return {
            "trace": current_trace_context.get(),
            "parent_span": current_span_context.get(),
        }

    def _get_current_span_safe(self) -> Optional[BaseSpan]:
        """
        Get the current span, trying ContextVar first then falling back to trace_manager.

        In async/threaded contexts, ContextVars may not propagate correctly.
        This method tries to find the current span by:
        1. First checking current_span_context (works when in same context)
        2. Falling back to looking up via current_trace_context's root spans
        """
        span = current_span_context.get()
        if span:
            return span

        # Fallback: Try to find the most recently active span from active_spans
        # that belongs to the current trace
        trace = current_trace_context.get()
        if trace:
            # Find spans belonging to this trace
            for span_uuid, active_span in trace_manager.active_spans.items():
                if active_span.trace_uuid == trace.uuid:
                    return active_span

        return None

    def _get_current_trace_safe(self) -> Optional[Trace]:
        """
        Get the current trace, trying ContextVar first then falling back to trace_manager.
        """
        trace = current_trace_context.get()
        if trace and trace.uuid in trace_manager.active_traces:
            return trace

        # Fallback: If there's only one active trace, use it
        if len(trace_manager.active_traces) == 1:
            return list(trace_manager.active_traces.values())[0]

        return None

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
        """Generate a unique ID for LLM call execution."""
        source_id = id(source)
        # Use from_agent and from_task if available
        agent_id = (
            id(event.from_agent)
            if hasattr(event, "from_agent") and event.from_agent
            else "unknown"
        )
        task_id = (
            id(event.from_task)
            if hasattr(event, "from_task") and event.from_task
            else "unknown"
        )
        model = getattr(event, "model", "unknown")
        # Include a hash of messages to differentiate multiple calls
        messages_hash = hash(str(getattr(event, "messages", "")))
        execution_id = (
            f"llm_{source_id}_{agent_id}_{task_id}_{model}_{messages_hash}"
        )

        return execution_id

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            # Use safe getters that work across context boundaries
            current_span = self._get_current_span_safe()
            current_trace = self._get_current_trace_safe()

            # set the input on span
            if current_span:
                current_span.input = event.inputs

            # set trace input
            if current_trace:
                current_trace.input = event.inputs

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            # Use safe getters that work across context boundaries
            current_span = self._get_current_span_safe()
            current_trace = self._get_current_trace_safe()

            # set the output on span
            if current_span:
                current_span.output = str(event.output)

            # set trace output
            if current_trace:
                current_trace.output = str(event.output)

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_started(source, event: LLMCallStartedEvent):
            # Capture current context before creating the observer
            context_data = self._capture_current_context()

            # Extract metric_collection from the parent span (agent span)
            # Note: event.from_agent is cleared by CrewAI's event processing,
            # so we get it from the parent span instead
            metric_collection = None
            parent_span = context_data.get("parent_span")
            if parent_span:
                metric_collection = getattr(
                    parent_span, "metric_collection", None
                )

            observer = Observer(
                span_type="llm",
                func_name="call",
                observe_kwargs={"model": event.model},
                metric_collection=metric_collection,
            )
            context_data["observer"] = observer

            execution_id = self.get_llm_execution_id(source, event)
            self.span_data[execution_id] = context_data
            observer.__enter__()

            # Set input on the newly created span
            span = trace_manager.get_span_by_uuid(observer.uuid)
            if span:
                span.input = event.messages
                if isinstance(span, LlmSpan):
                    span.model = event.model

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_completed(source, event: LLMCallCompletedEvent):
            execution_id = self.get_llm_execution_id(source, event)
            stored_data = self.span_data.pop(execution_id, None)

            if stored_data:
                observer = stored_data.get("observer")
                if observer:
                    # Restore context before exiting
                    with self._restore_context(stored_data):
                        # Set output on the span directly via trace_manager lookup
                        span = trace_manager.get_span_by_uuid(observer.uuid)
                        if span:
                            span.output = event.response
                        observer.__exit__(None, None, None)

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_started(source, event: AgentExecutionStartedEvent):
            # Use safe getter that works across context boundaries
            current_span = self._get_current_span_safe()

            # set the input
            if current_span:
                current_span.input = event.task_prompt

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_completed(source, event: AgentExecutionCompletedEvent):
            # Use safe getter that works across context boundaries
            current_span = self._get_current_span_safe()

            # set the output
            if current_span:
                current_span.output = event.output

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_started(source, event: ToolUsageStartedEvent):
            # Capture current context before creating the observer
            context_data = self._capture_current_context()

            # Extract metric_collection from parent span
            metric_collection = None
            parent_span = context_data.get("parent_span")
            if parent_span:
                metric_collection = getattr(
                    parent_span, "metric_collection", None
                )

            observer = Observer(
                span_type="tool",
                func_name=event.tool_name,
                function_kwargs=event.tool_args,
                metric_collection=metric_collection,
            )
            context_data["observer"] = observer

            execution_id = self.get_tool_execution_id(source, event)
            self.span_data[execution_id] = context_data
            observer.__enter__()

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_completed(source, event: ToolUsageFinishedEvent):
            execution_id = self.get_tool_execution_id(source, event)
            stored_data = self.span_data.pop(execution_id, None)

            if stored_data:
                observer = stored_data.get("observer")
                if observer:
                    # Restore context before exiting - this ensures __exit__
                    # sees the correct span in current_span_context
                    with self._restore_context(stored_data):
                        # Set output on the span directly via trace_manager lookup
                        span = trace_manager.get_span_by_uuid(observer.uuid)
                        if span:
                            span.output = event.output
                        observer.__exit__(None, None, None)

        @crewai_event_bus.on(KnowledgeRetrievalStartedEvent)
        def on_knowledge_started(source, event: KnowledgeRetrievalStartedEvent):
            # Capture current context before creating the observer
            context_data = self._capture_current_context()

            # Extract metric_collection from parent span
            metric_collection = None
            parent_span = context_data.get("parent_span")
            if parent_span:
                metric_collection = getattr(
                    parent_span, "metric_collection", None
                )

            observer = Observer(
                span_type="tool",
                func_name="knowledge_retrieval",
                function_kwargs={},
                metric_collection=metric_collection,
            )
            context_data["observer"] = observer

            execution_id = self.get_knowledge_execution_id(source, event)
            self.span_data[execution_id] = context_data
            observer.__enter__()

        @crewai_event_bus.on(KnowledgeRetrievalCompletedEvent)
        def on_knowledge_completed(
            source, event: KnowledgeRetrievalCompletedEvent
        ):
            execution_id = self.get_knowledge_execution_id(source, event)
            stored_data = self.span_data.pop(execution_id, None)

            if stored_data:
                observer = stored_data.get("observer")
                if observer:
                    # Restore context before exiting
                    with self._restore_context(stored_data):
                        # Set input/output on the span directly via trace_manager lookup
                        span = trace_manager.get_span_by_uuid(observer.uuid)
                        if span:
                            span.input = event.query
                            span.output = event.retrieved_knowledge
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
