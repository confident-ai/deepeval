"""
PydanticAI Streaming App: Agent with streaming response
Complexity: MEDIUM - Tests streaming functionality

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings


def create_streaming_agent(
    name: str = "pydanticai-streaming-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
) -> Agent:
    """Create a PydanticAI agent for streaming with instrumentation settings."""
    settings = ConfidentInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "streaming"],
        metadata=metadata or {"test_type": "streaming"},
        thread_id=thread_id,
        user_id=user_id,
        is_test_mode=True,
    )

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one short sentence only.",
        instrument=settings,
        name="streaming_agent",
    )


async def stream_agent(prompt: str, agent: Agent = None) -> str:
    """Invoke the agent with streaming and collect the full response."""
    if agent is None:
        agent = create_streaming_agent()

    full_response = ""
    async with agent.run_stream(prompt) as response:
        async for chunk in response.stream_text():
            full_response += chunk

    return full_response
