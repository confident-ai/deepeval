"""A Lightweight, thread safe registry for passing ContextVars
Contexts between CrewAI wrapper code and event bus listeners.

CrewAI often emits events on worker threads that do not inherit the
caller's ContextVars state. We capture a Context snapshot, via
contextvars.copy_context, at the wrapper boundary and stash it here,
keyed by a stable runtime identifier. Event listeners then look up and
re-bind that exact Context before entering or exiting spans, ensuring
correct parent/trace linkage even across threads.
"""

from contextvars import Context
from threading import RLock


class _Table:
    """A tiny, thread safe mapping from keys to Context objects.

    Uses an RLock because the same thread may re-enter while handling
    nested events.
    """

    def __init__(self):
        self._entries = {}
        self._lock = RLock()

    def set(self, key, value: Context):
        with self._lock:
            self._entries[key] = value

    def get(self, key, default=None):
        with self._lock:
            return self._entries.get(key, default)

    def pop(self, key, default=None):
        with self._lock:
            return self._entries.pop(key, default)


class ContextRegistry:
    """Namespaced tables for Context snapshots used by integrations.

    Today we only need an `agent` table to support the CrewAI agent
    execute lifecycle, but the namespace makes it easy to add `tool`,
    `crew`, etc., later without mixing keys.
    """

    def __init__(self):
        self.agent = _Table()


CONTEXT_REG = ContextRegistry()
