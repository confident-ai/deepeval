"""
PydanticAI Prompt App: Agent with Confident Prompt logging
Complexity: LOW - Tests prompt attribution feature

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings
from deepeval.prompt import Prompt


def create_prompt_agent(
    prompt_alias: str = "test-prompt",
    prompt_version: str = "00.00.01",
    name: str = "pydanticai-prompt-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
) -> Agent:
    """Create a PydanticAI agent with prompt logging.

    Note: We create a Prompt object but don't pull it (no network call).
    The prompt alias and version are logged for attribution purposes.
    """
    # Create prompt object for attribution (without pulling)
    prompt = Prompt(alias=prompt_alias)
    prompt.hash = "bab4ce0"
    prompt.version = prompt_version
    prompt.label = "test-prompt-label"

    settings = ConfidentInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "prompt"],
        metadata=metadata or {"test_type": "prompt"},
        thread_id=thread_id,
        user_id=user_id,
        confident_prompt=prompt,
        is_test_mode=True,
    )

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one short sentence only.",
        instrument=settings,
        name="prompt_agent",
    )


def invoke_prompt_agent(prompt: str, agent: Agent = None) -> str:
    """Invoke the prompt agent synchronously."""
    if agent is None:
        agent = create_prompt_agent()
    result = agent.run_sync(prompt)
    return result.output


async def ainvoke_prompt_agent(prompt: str, agent: Agent = None) -> str:
    """Invoke the prompt agent asynchronously."""
    if agent is None:
        agent = create_prompt_agent()
    result = await agent.run(prompt)
    return result.output
