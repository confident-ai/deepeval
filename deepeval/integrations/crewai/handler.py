import contextvars
import deepeval
import json
import logging
from typing import Optional

from deepeval.utils import should_async_debug
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.context import current_span_context, current_trace_context
from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.tracing.types import BaseSpan, LlmSpan, AgentSpan
from .subs import (
    validate_crewai_installed,
    is_crewai_installed,
    try_import_events,
)
from deepeval.tracing._debug import (
    print_dbg_tag,
    _task_id,
    _ctx_id,
    _thread_name,
)
from .context_registration import CONTEXT_REG
from .identifiers import agent_exec_id


logger = logging.getLogger(__name__)

# Keep a strong module-level reference so the listener, and its event
# subscriptions, can’t be garbage collected.
CREWAI_LISTENER = None

(
    BaseEventListener,
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
) = try_import_events()

crewai_installed = is_crewai_installed()
IS_WRAPPED_ALL = False

if BaseEventListener is not None:

    class CrewAIEventsListener(BaseEventListener):
        """Bridges CrewAI’s event bus to DeepEval spans (thread-safe).

        Opens/closes Agent/LLM/Tool/Knowledge spans on start/finish events, while
        restoring the exact ContextVars snapshot captured at wrapper boundaries so
        parent/trace linkage remains correct across worker threads. Uses:
          - `span_observers` to manage active Observer contexts by exec id.
          - `_exec_diag` to stash execution context and minimal diagnostics.

        Note: Enable diagnostic logging via the following settings:
          - LOG_LEVEL = "debug"
          - DEEPEVAL_DEBUG_ASYNC = True
        see deepeval/trace/_debug.py for more details on diagnostics
        """

        def __init__(self):
            validate_crewai_installed()
            super().__init__()
            self.span_observers: dict[str, Observer] = {}

            # diagnostics
            self._exec_diag: dict[str, dict] = {}

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
                exec_id = self.get_tool_execution_id(source, event)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.KickoffStart",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )

                ctx = contextvars.copy_context()
                # keep ctx so completed can reuse it
                self._exec_diag[exec_id] = {"ctx": ctx}

                def _apply_inputs():
                    root_span = self._find_active_root_span()
                    if root_span:
                        # bind both span and trace into this ctx
                        current_span_context.set(root_span)
                        root_trace = trace_manager.active_traces.get(
                            root_span.trace_uuid
                        )
                        if root_trace:
                            current_trace_context.set(root_trace)

                    # Assuming that this event is called in the crew.kickoff method
                    current_span = current_span_context.get()

                    # set the input
                    if current_span:
                        current_span.input = event.inputs

                    # set trace input
                    current_trace = current_trace_context.get()
                    if current_trace:
                        current_trace.input = event.inputs

                    self._exec_diag[exec_id]["ctx"] = contextvars.copy_context()

                    # fallback to lookin up the root span
                    if not (current_span and current_trace):
                        root_span = self._find_active_root_span()
                        if root_span:
                            root_span.input = event.inputs or {}
                            current_trace = trace_manager.active_traces.get(
                                root_span.trace_uuid
                            )
                            if current_trace:
                                current_trace.input = event.inputs or {}

                trace_manager.run_span_op(_apply_inputs, ctx)

            @crewai_event_bus.on(CrewKickoffCompletedEvent)
            def on_crew_completed(source, event: CrewKickoffCompletedEvent):
                exec_id = self.get_tool_execution_id(source, event)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.KickoffDone",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )

                ctx = (self._exec_diag.pop(exec_id, None) or {}).get("ctx")
                ctx = ctx or contextvars.copy_context()

                def _apply_outputs():
                    # Assuming that this event is called in the crew.kickoff method
                    current_span = current_span_context.get()
                    # set the output
                    if current_span:
                        current_span.output = str(event.output)

                    # set trace output
                    current_trace = current_trace_context.get()
                    if current_trace:
                        current_trace.output = str(event.output)

                    # fallback to looking up the root span
                    if not (current_span and current_trace):
                        root_span = self._find_active_root_span()
                        if root_span:
                            out = str(event.output)
                            root_span.output = out
                            current_trace = trace_manager.active_traces.get(
                                root_span.trace_uuid
                            )
                            if current_trace:
                                current_trace.output = out

                trace_manager.run_span_op(_apply_outputs, ctx)

            @crewai_event_bus.on(LLMCallStartedEvent)
            def on_llm_started(source, event: LLMCallStartedEvent):
                exec_id = self.get_tool_execution_id(source, event)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.LLMStart",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )

                ctx = contextvars.copy_context()
                self._exec_diag[exec_id] = {"ctx": ctx}
                observer = Observer(span_type="llm", func_name="call")
                self.span_observers[exec_id] = observer

                def _apply_llm_start():
                    parent = self._find_active_agent_span()
                    if parent is not None:
                        current_span_context.set(parent)
                        current_trace = trace_manager.active_traces.get(
                            parent.trace_uuid
                        )
                        if current_trace:
                            current_trace_context.set(current_trace)

                    observer.__enter__()

                    # capture worker ctx for exit
                    self._exec_diag[exec_id]["ctx"] = contextvars.copy_context()

                    # fill LLM span fields right away
                    current_span = current_span_context.get()
                    if isinstance(current_span, LlmSpan):
                        current_span.model = getattr(event, "model", None)
                        current_span.input = event.messages

                trace_manager.run_span_op(_apply_llm_start, ctx)

            @crewai_event_bus.on(LLMCallCompletedEvent)
            def on_llm_completed(source, event: LLMCallCompletedEvent):
                exec_id = self.get_tool_execution_id(source, event)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.LLMDone",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )

                observer = self.span_observers.pop(exec_id, None)
                ctx = (self._exec_diag.pop(exec_id, None) or {}).get("ctx")
                ctx = ctx or contextvars.copy_context()

                if observer:

                    def _apply_llm_done():
                        # Assuming that this event is called in the llm.call method
                        current_span = current_span_context.get()

                        # set the output
                        if current_span:
                            current_span.output = event.response
                        observer.__exit__(None, None, None)

                    trace_manager.run_span_op(_apply_llm_done, ctx)

            @crewai_event_bus.on(AgentExecutionStartedEvent)
            def on_agent_started(source, event: AgentExecutionStartedEvent):
                """Agent started, enrich the agent span with input in the correct context.

                Why do we use CONTEXT_REG to get the agent context?:
                - CrewAI can emit this event on a worker thread whose ContextVars don't include
                  the span opened in the agent wrapper.
                - We therefore restore the exact Context captured in `wrap_agent_execute_task`
                  (see `CONTEXT_REG.agent.set(agent_exec_id(self), copy_context())`) so that
                  the current thread has the proper parent/trace linkage before we touch spans.

                See context_registration.py for how CONTEXT_REG works.
                """
                # Use a agent_exec_id, our wrapper stores the context there
                exec_id = agent_exec_id(source)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.AgentStart",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )
                # Restore the wrapper captured Context, otherwise default to the current thread’s context.
                # This Context is set in `wrap_agent_execute_task` (wrapper.py).
                ctx = (
                    CONTEXT_REG.agent.get(exec_id) or contextvars.copy_context()
                )
                self._exec_diag[exec_id] = {"ctx": ctx}

                def _apply_agent_start():
                    # enrich Agent span input in the correct context
                    current_span = current_span_context.get()
                    if current_span and current_span.input in (None, {}):
                        try:
                            current_span.input = event.task_prompt
                        except Exception:
                            pass

                trace_manager.run_span_op(_apply_agent_start, ctx)

            @crewai_event_bus.on(AgentExecutionCompletedEvent)
            def on_agent_completed(source, event: AgentExecutionCompletedEvent):
                """On agent done, finalize the agent span’s output in the same Context where it was opened.

                See context_registration.py for how CONTEXT_REG works.
                """
                exec_id = agent_exec_id(source)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.AgentDone",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )
                # Prefer the Context set in the wrapper, pop it now to avoid leaks.
                # If that’s missing, then fall back to the one captured in the start handler.
                # This Context was set in `wrap_agent_execute_task` (wrapper.py).
                ctx = (
                    CONTEXT_REG.agent.pop(exec_id, None)
                    or (self._exec_diag.pop(exec_id, None) or {}).get("ctx")
                    or contextvars.copy_context()
                )

                def _apply_agent_done():
                    # enrich. Wrapper already set output, so this only needs to fill gaps.
                    current_span = current_span_context.get()
                    if current_span and not current_span.output:
                        current_span.output = event.output
                    current_trace = current_trace_context.get()
                    if current_trace and not current_trace.output:
                        current_trace.output = event.output

                trace_manager.run_span_op(_apply_agent_done, ctx)

            @crewai_event_bus.on(ToolUsageStartedEvent)
            def on_tool_started(source, event: ToolUsageStartedEvent):
                exec_id = self.get_tool_execution_id(source, event)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.ToolStart",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )
                ctx = contextvars.copy_context()

                observer = Observer(
                    span_type="tool",
                    func_name=event.tool_name,
                    function_kwargs=event.tool_args,
                )
                self.span_observers[exec_id] = observer

                self._exec_diag[exec_id] = {
                    "schedule_thr": _thread_name(),
                    "schedule_task_id": _task_id(),
                    "schedule_ctx_id": _ctx_id(),
                    "enter_thr": None,
                    "enter_task": None,
                    "enter_ctx_id": None,
                    "ctx": None,
                    "observer_uuid": observer.uuid,
                }

                parent = self._find_active_agent_span()

                def _apply_tool_enter():
                    # ensure parent and trace are set for __enter__
                    if parent is not None:
                        current_span_context.set(parent)
                        tr = trace_manager.active_traces.get(parent.trace_uuid)
                        if tr:
                            current_trace_context.set(tr)
                    # capture worker ctx and record
                    diag = self._exec_diag.get(exec_id)
                    if diag is not None:
                        diag["enter_thr"] = _thread_name()
                        diag["enter_task"] = _task_id()
                        diag["enter_ctx_id"] = _ctx_id()

                    observer.__enter__()

                    # after __enter__, capture the worker Context we’ll use for __exit__
                    worker_ctx = contextvars.copy_context()
                    if diag is not None:
                        diag["ctx"] = worker_ctx

                    # set tool input explicitly
                    try:
                        current_span = current_span_context.get()
                        if current_span:
                            current_span.input = json.dumps(
                                event.tool_args or {}
                            )
                    except Exception:
                        pass

                trace_manager.run_span_op(_apply_tool_enter, ctx)

            @crewai_event_bus.on(ToolUsageFinishedEvent)
            def on_tool_completed(source, event: ToolUsageFinishedEvent):
                exec_id = self.get_tool_execution_id(source, event)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.ToolDone",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )

                observer = self.span_observers.pop(exec_id, None)

                diag = self._exec_diag.pop(exec_id, None) or {}
                ctx = diag.get("ctx") or contextvars.copy_context()

                # Log diagnostics if the worker enter thread ctx differs from the finish thread ctx
                if diag.get("enter_thr") is not None:
                    same_thr = diag["enter_thr"] == _thread_name()
                    same_ctx = diag["enter_ctx_id"] == _ctx_id()
                    same_task = diag["enter_task"] == _task_id()
                    if not (same_thr and same_ctx and same_task):
                        if should_async_debug():
                            print_dbg_tag(
                                "EVT.ToolPairCrossThread",
                                extra=(
                                    f"exec={exec_id} "
                                    f"enter_thr={diag['enter_thr']} -> end_thr={_thread_name()} "
                                    f"enter_ctx={diag['enter_ctx_id']} -> end_ctx={_ctx_id()} "
                                    f"enter_task={diag['enter_task']} -> end_task={_task_id()}"
                                ),
                            )

                if observer:

                    def _apply_tool_exit():
                        cur = current_span_context.get()
                        if cur:
                            cur.output = event.output
                        observer.__exit__(None, None, None)

                    trace_manager.run_span_op(_apply_tool_exit, ctx)

            @crewai_event_bus.on(KnowledgeRetrievalStartedEvent)
            def on_knowledge_started(
                source, event: KnowledgeRetrievalStartedEvent
            ):
                exec_id = self.get_knowledge_execution_id(source, event)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.KnowledgeStart",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )

                ctx = contextvars.copy_context()
                observer = Observer(
                    span_type="tool",
                    func_name="knowledge_retrieval",
                    function_kwargs={},
                )
                self.span_observers[exec_id] = observer

                self._exec_diag[exec_id] = {
                    "enter_thr": None,
                    "enter_task": None,
                    "enter_ctx_id": None,
                    "ctx": None,
                    "observer_uuid": observer.uuid,
                }

                parent = self._find_active_agent_span()

                def _apply_knowledge_start():
                    if parent is not None:
                        current_span_context.set(parent)
                        current_trace = trace_manager.active_traces.get(
                            parent.trace_uuid
                        )
                        if current_trace:
                            current_trace_context.set(current_trace)

                    observer.__enter__()

                    worker_ctx = contextvars.copy_context()
                    diag = self._exec_diag.get(exec_id)
                    if diag is not None:
                        diag["enter_thr"] = _thread_name()
                        diag["enter_task"] = _task_id()
                        diag["enter_ctx_id"] = _ctx_id()
                        diag["ctx"] = worker_ctx

                trace_manager.run_span_op(_apply_knowledge_start, ctx)

            @crewai_event_bus.on(KnowledgeRetrievalCompletedEvent)
            def on_knowledge_completed(
                source, event: KnowledgeRetrievalCompletedEvent
            ):
                exec_id = self.get_knowledge_execution_id(source, event)
                if should_async_debug():
                    print_dbg_tag(
                        "EVT.KnowledgeDone",
                        extra=f"thr={_thread_name()} task={_task_id()} ctx={_ctx_id()} exec={exec_id}",
                    )

                observer = self.span_observers.pop(exec_id, None)
                diag = self._exec_diag.pop(exec_id, None) or {}
                ctx = diag.get("ctx") or contextvars.copy_context()

                if observer:

                    def _apply_knowledge_done():
                        cur = current_span_context.get()
                        if cur:
                            cur.input = event.query
                            cur.output = event.retrieved_knowledge
                        observer.__exit__(None, None, None)

                    trace_manager.run_span_op(_apply_knowledge_done, ctx)

        def _find_active_root_span(self):
            # root = base span that hasn't finished yet
            for span in trace_manager.active_spans.values():
                if (
                    isinstance(span, BaseSpan)
                    and span.parent_uuid is None
                    and span.end_time is None
                ):
                    return span
            return None

        def _find_active_agent_span(self):
            for span in trace_manager.active_spans.values():
                if isinstance(span, AgentSpan) and span.end_time is None:
                    return span
            return None

else:
    CrewAIEventsListener = None


def instrument_crewai(api_key: Optional[str] = None):
    """Install DeepEval CrewAI tracing.

    - Verifies CrewAI is available and no-ops with a warning if not.
    - Optionally logs in to DeepEval if `api_key` is provided.
    - Wraps key CrewAI entry points to open root spans.
    - Attaches a singleton event listener that mirrors CrewAI events into spans.
    Safe to call multiple times, the listener is created once and retained globally.
    """
    # Only proceed when the listener is available, this avoids import-time exceptions on incompatible environments
    if not crewai_installed or CrewAIEventsListener is None:
        logger.warning(
            "CrewAI events listener is unavailable. Skipping CrewAI instrumentation."
        )
        return
    with capture_tracing_integration("crewai"):
        if api_key:
            deepeval.login(api_key)

        wrap_all()

        # module bound to protect from garbage collector
        global CREWAI_LISTENER
        if CREWAI_LISTENER is None:
            CREWAI_LISTENER = CrewAIEventsListener()


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
