"""
PydanticAI Evals App: Comprehensive trace-level features.
Complexity: MEDIUM - Tests trace-level metadata + tool spans.

After the settings refactor, ``DeepEvalInstrumentationSettings`` carries
ONLY trace-level defaults (``name``, ``thread_id``, ``user_id``, ``tags``,
``metadata``, ``metric_collection``, ``test_case_id``, ``turn_id``).
Per-span configuration is set at runtime — either by ``update_current_*_span(...)``
from inside the body of a span the user owns, or by ``next_*_span(...)``
context managers wrapping the agent call for spans the user can't enter
(agent / LLM spans emitted by pydantic-ai itself).

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from typing import Dict, List, Optional
from pydantic_ai import Agent

from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings
from deepeval.tracing import next_agent_span


def create_evals_agent(
    metric_collection: Optional[str] = None,
    name: str = "pydanticai-evals-test",
    tags: List[str] = None,
    metadata: Dict = None,
    thread_id: str = None,
    user_id: str = None,
) -> Agent:
    """Create a PydanticAI agent with trace-level instrumentation settings."""

    settings = DeepEvalInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "evals"],
        metadata=metadata or {"test_type": "evals"},
        thread_id=thread_id,
        user_id=user_id,
        metric_collection=metric_collection,
    )

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Be concise.",
        instrument=settings,
        name="evals_agent",
    )

    @agent.tool_plain
    def special_tool(query: str) -> str:
        """A tool used by feature tests."""
        return f"Processed: {query}"

    return agent


def invoke_evals_agent(
    prompt: str,
    agent: Agent,
    agent_metric_collection: Optional[str] = None,
) -> str:
    """Invoke the evals agent synchronously.

    ``agent_metric_collection`` (if provided) is staged via
    ``next_agent_span(metric_collection=...)`` so it lands on the
    pydantic-ai-emitted agent span — replacing the dropped
    ``settings.agent_metric_collection`` kwarg. The user can't reach
    inside the agent span body to call ``update_current_span(...)``,
    so the wrapper-staging path is the only mechanism."""
    if agent_metric_collection:
        with next_agent_span(metric_collection=agent_metric_collection):
            return agent.run_sync(prompt).output
    return agent.run_sync(prompt).output


async def ainvoke_evals_agent(
    prompt: str,
    agent: Agent,
    agent_metric_collection: Optional[str] = None,
) -> str:
    """Async equivalent of ``invoke_evals_agent``."""
    if agent_metric_collection:
        with next_agent_span(metric_collection=agent_metric_collection):
            result = await agent.run(prompt)
            return result.output
    result = await agent.run(prompt)
    return result.output
