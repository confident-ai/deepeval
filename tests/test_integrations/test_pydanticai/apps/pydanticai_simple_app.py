"""
Simple PydanticAI App: LLM-only, no tools
Complexity: LOW - Tests basic agent invocation

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings


def create_simple_agent(
    name: str = "pydanticai-simple-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
) -> Agent:
    """Create a simple PydanticAI agent with instrumentation settings."""
    settings = ConfidentInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "simple"],
        metadata=metadata or {"test_type": "simple"},
        thread_id=thread_id,
        user_id=user_id,
        is_test_mode=True,
    )

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one short sentence only.",
        instrument=settings,
        name="simple_agent",
    )


def invoke_simple_agent(prompt: str, agent: Agent = None) -> str:
    """Invoke the simple agent synchronously."""
    if agent is None:
        agent = create_simple_agent()
    result = agent.run_sync(prompt)
    return result.output


async def ainvoke_simple_agent(prompt: str, agent: Agent = None) -> str:
    """Invoke the simple agent asynchronously."""
    if agent is None:
        agent = create_simple_agent()
    result = await agent.run(prompt)
    return result.output
