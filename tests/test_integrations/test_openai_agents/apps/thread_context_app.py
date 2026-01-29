"""
tests/test_integrations/test_openai_agents/apps/thread_context_app.py
Tests persistence of thread_id and user_id in traces.
"""

from deepeval.openai_agents import Agent


def get_thread_context_app():
    """
    Returns an agent for testing trace attributes.
    """
    agent = Agent(
        name="Context Aware Bot",
        instructions="You are a helpful assistant. Just say 'Acknowledged'.",
    )

    return agent, "Session check."
