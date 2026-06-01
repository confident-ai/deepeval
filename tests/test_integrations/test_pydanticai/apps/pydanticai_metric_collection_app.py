"""
PydanticAI Metric Collection App: Agent with trace-level metric collection.
Complexity: LOW - Tests trace-level online evaluation metric collection.

Trace-level ``metric_collection`` is set via ``DeepEvalInstrumentationSettings``
(it's a trace default, alongside ``name`` / ``tags`` / ``user_id`` / etc.).
It can also be overridden at runtime from anywhere in the call stack via
``update_current_trace(metric_collection=...)`` — the runtime value wins.

Per-span ``metric_collection`` is no longer a settings concern. Use
``update_current_span(metric_collection=...)`` from inside your tool /
agent body for spans you own.

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from typing import Optional

from pydantic_ai import Agent

from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings


def create_trace_metric_collection_agent(
    metric_collection: Optional[str] = None,
    name: str = "pydanticai-trace-metric-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
) -> Agent:
    """Create a PydanticAI agent with trace-level ``metric_collection``."""
    settings = DeepEvalInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "trace-metric-collection"],
        metadata=metadata or {"test_type": "trace_metric_collection"},
        thread_id=thread_id,
        user_id=user_id,
        metric_collection=metric_collection,
    )

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one short sentence only.",
        instrument=settings,
        name="trace_metric_agent",
    )


def invoke_metric_collection_agent(prompt: str, agent: Agent) -> str:
    """Invoke the agent synchronously."""
    return agent.run_sync(prompt).output


async def ainvoke_metric_collection_agent(prompt: str, agent: Agent) -> str:
    """Async equivalent of ``invoke_metric_collection_agent``."""
    result = await agent.run(prompt)
    return result.output
