"""
PydanticAI Metric Collection App: Agent with metric collection settings
Complexity: LOW - Tests online evaluation metric collection attributes

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings


def create_trace_metric_collection_agent(
    metric_collection: str = "test-trace-metrics",
    name: str = "pydanticai-trace-metric-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
) -> Agent:
    """Create a PydanticAI agent with trace-level metric collection."""
    settings = ConfidentInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "trace-metric-collection"],
        metadata=metadata or {"test_type": "trace_metric_collection"},
        thread_id=thread_id,
        user_id=user_id,
        trace_metric_collection=metric_collection,
        is_test_mode=True,
    )

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one short sentence only.",
        instrument=settings,
        name="trace_metric_agent",
    )


def create_agent_metric_collection_agent(
    metric_collection: str = "test-agent-metrics",
    name: str = "pydanticai-agent-metric-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
) -> Agent:
    """Create a PydanticAI agent with agent-span-level metric collection."""
    settings = ConfidentInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "agent-metric-collection"],
        metadata=metadata or {"test_type": "agent_metric_collection"},
        thread_id=thread_id,
        user_id=user_id,
        agent_metric_collection=metric_collection,
        is_test_mode=True,
    )

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one short sentence only.",
        instrument=settings,
        name="agent_metric_agent",
    )


def create_llm_metric_collection_agent(
    metric_collection: str = "test-llm-metrics",
    name: str = "pydanticai-llm-metric-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
) -> Agent:
    """Create a PydanticAI agent with LLM-span-level metric collection."""
    settings = ConfidentInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "llm-metric-collection"],
        metadata=metadata or {"test_type": "llm_metric_collection"},
        thread_id=thread_id,
        user_id=user_id,
        llm_metric_collection=metric_collection,
        is_test_mode=True,
    )

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one short sentence only.",
        instrument=settings,
        name="llm_metric_agent",
    )


def invoke_metric_collection_agent(prompt: str, agent: Agent) -> str:
    """Invoke any metric collection agent synchronously."""
    result = agent.run_sync(prompt)
    return result.output


async def ainvoke_metric_collection_agent(prompt: str, agent: Agent) -> str:
    """Invoke any metric collection agent asynchronously."""
    result = await agent.run(prompt)
    return result.output
