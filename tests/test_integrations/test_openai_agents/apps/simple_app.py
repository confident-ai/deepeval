"""
tests/test_integrations/test_openai_agents/apps/simple_app.py
A basic single-agent setup for testing simple execution traces.
"""

from deepeval.openai_agents import Agent


def get_simple_app():
    """
    Returns a simple agent that echoes back greetings.
    """
    agent = Agent(
        name="Simple Greeter",
        instructions="You are a friendly bot. Reply to greetings with 'Hello World'.",
    )

    return agent, "Hi there!"
