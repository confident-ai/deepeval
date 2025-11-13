"""Stable, human readable runtime identifiers for CrewAI objects.

Keeping these in a tiny module avoids import cycle risks and makes it easy
to extend while centralizing the ID format.
"""


def agent_exec_id(agent_obj) -> str:
    """Return a stable identifier for this agent instance within the
    process.

    Uses Python's object identity `id()` with an `agent_` prefix to
    avoid namespace collisions. This is sufficient for correlating the
    wrapper captured Context (see wrap_agent_execute_task) with event
    bus callbacks during a single run.
    """
    return f"agent_{id(agent_obj)}"
