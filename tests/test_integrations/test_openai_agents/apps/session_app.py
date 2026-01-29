"""
tests/test_integrations/test_openai_agents/apps/session_app.py
Tests multi-turn conversation memory using SQLiteSession.
"""

from deepeval.openai_agents import Agent

try:
    from agents import SQLiteSession
except ImportError:
    SQLiteSession = None


def get_session_app(session_id: str = "test_session_123"):
    """
    Returns an agent and a Session object to test memory persistence.
    """
    if not SQLiteSession:
        return None, None, None

    agent = Agent(
        name="Memory Bot",
        instructions="You have a memory. Remember what the user tells you.",
    )

    # Use an in-memory SQLite database for testing to avoid file cleanup issues
    session = SQLiteSession(session_id, ":memory:")

    # We return a list of inputs to simulate a conversation
    inputs = [
        "My favorite color is Blue.",
        "What is my favorite color?",
    ]

    return agent, session, inputs
