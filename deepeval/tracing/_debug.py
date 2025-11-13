"""Debug helpers for tracing that are inert unless explicitly enabled.

All emission is gated by:
  1. logger level (DEBUG), and
  2. get_settings().DEEPEVAL_DEBUG_ASYNC == True

This lets developers turn on rich, cross-thread diagnostics (thread
name, task id, Context id) without polluting stdout in normal
runs. Prefer these helpers over ad-hoc prints so users can redirect
via logging configuration.  """

import asyncio
import contextvars
import importlib
import inspect
import threading
import logging


from deepeval.config.settings import get_settings


logger = logging.getLogger(__name__)


def _dbg(msg: str):
    """Emit a DEBUG log line if async debug is enabled."""
    if (
        logger.isEnabledFor(logging.DEBUG)
        and get_settings().DEEPEVAL_DEBUG_ASYNC
    ):
        logger.debug(msg)


def dbg_tag(label, extra=None):
    """Build a standardized tag with thread/task/context IDs.

    Note: ContextVars are local to a single context and effectively
    local to the current thread and async task, so including the
    Context id is a big help when diagnosing cross-thread event
    handling.
    """
    try:
        task = asyncio.current_task()
        task_id = (
            f"{id(task)}:{getattr(task, 'get_name', lambda: 'n/a')()}"
            if task
            else "None"
        )
    except Exception:
        task_id = "n/a"
    ctx_id = id(contextvars.copy_context())
    thr = threading.current_thread().name
    return (
        f"[{label}] thr={thr} task={task_id} ctx={ctx_id} extra={extra or ''}"
    )


def print_dbg_tag(label, extra=None):
    """Compute and emit a dbg_tag in one call."""
    _dbg(dbg_tag(label, extra))


def _task_id():
    """Return a hex id for the current asyncio task or 'n/a'."""
    try:
        task = asyncio.current_task()
    except Exception:
        task = None
    return hex(id(task)) if task else "n/a"


def _ctx_id():
    """Return a hex id for the current ContextVars Context snapshot."""
    return hex(id(contextvars.copy_context()))


def _thread_name():
    """Return the current thread's name."""
    return threading.current_thread().name


################
# Integrations #
################


def _debug_dump_crewai_events():
    """Log available CrewAI event symbols and listener shape.

    Useful for quickly verifying which events exist in the installed CrewAI
    version and whether BaseEventListener exposes the hooks we rely on.
    """
    try:
        events_mod = importlib.import_module("crewai.events")
        _dbg(
            "[crewai.events] symbols:",
            [
                name
                for name in dir(events_mod)
                if "Event" in name or name == "BaseEventListener"
            ],
        )
        bel = getattr(events_mod, "BaseEventListener", None)
        if bel:
            _dbg(
                "BaseEventListener.init signature:",
                inspect.signature(bel.__init__),
            )
            _dbg(
                "BaseEventListener has setup_listeners:",
                hasattr(bel, "setup_listeners"),
            )
    except Exception as e:
        _dbg("FAILED to import crewai.events:", repr(e))
