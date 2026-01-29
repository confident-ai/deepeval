"""
tests/test_integrations/test_openai_agents/apps/streaming_app.py
An agent designed for streaming execution tests.
"""

from deepeval.openai_agents import Agent


def get_streaming_app():
    """
    Returns an agent prompt to generate a longer response for streaming tests.
    """
    agent = Agent(
        name="Poet",
        instructions="You are a poet. Write a short 4-line poem about coding.",
    )

    return agent, "Write me a poem."
