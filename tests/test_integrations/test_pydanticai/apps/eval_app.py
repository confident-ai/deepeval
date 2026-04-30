"""
PydanticAI Evals App: Comprehensive testing of DeepEval features.
Complexity: MEDIUM - Tests metadata, metrics, and context injection.

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from typing import List, Dict, Optional
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings
from deepeval.prompt import Prompt
from deepeval.metrics import BaseMetric


def create_evals_agent(
    name: str = "pydanticai-evals-test",
    tags: List[str] = None,
    metadata: Dict = None,
    thread_id: str = None,
    user_id: str = None,
    metric_collection: str = None,
    agent_metric_collection: str = None,
    llm_metric_collection: str = None,
    tool_metric_collection_map: Dict = None,
    trace_metric_collection: str = None,
    confident_prompt: Prompt = None,
    agent_metrics: List[BaseMetric] = None,
) -> Agent:
    """Create a PydanticAI agent with full DeepEval instrumentation settings."""

    settings = ConfidentInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "evals"],
        metadata=metadata or {"test_type": "evals"},
        thread_id=thread_id,
        user_id=user_id,
        metric_collection=metric_collection,
        agent_metric_collection=agent_metric_collection,
        llm_metric_collection=llm_metric_collection,
        tool_metric_collection_map=tool_metric_collection_map,
        trace_metric_collection=trace_metric_collection,
        confident_prompt=confident_prompt,
        agent_metrics=agent_metrics,
        is_test_mode=True,
    )

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Be concise.",
        instrument=settings,
        name="evals_agent",
    )

    @agent.tool_plain
    def special_tool(query: str) -> str:
        """A tool to test tool metric collections."""
        return f"Processed: {query}"

    return agent


def invoke_evals_agent(prompt: str, agent: Agent) -> str:
    """Invoke the evals agent synchronously."""
    result = agent.run_sync(prompt)
    return result.output


async def ainvoke_evals_agent(prompt: str, agent: Agent) -> str:
    """Invoke the evals agent asynchronously."""
    result = await agent.run(prompt)
    return result.output
