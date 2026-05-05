"""PydanticAI Next-Span App: validates ``with next_llm_span(...)`` and
stacked ``with next_agent_span(...), next_llm_span(...)`` patterns.

Closes the schema-test coverage gap for ``next_llm_span`` —
``next_agent_span`` is exercised by ``eval_app.py`` / ``features_*.json``,
but the LLM-span staging slot had no end-to-end shape assertion despite
being the **only** mechanism by which a user can stamp LLM-span fields
(LLM spans are framework internals — no user-code seam).

Mirrors scenarios 1 and 2 from ``pydantic_after_next_span.py``. Scenarios
3 (one-shot consumption) and 4 (nested overrides) are NOT covered here:
they need 2 ``agent.run`` calls per test, but ``trace_testing_manager``
captures a single trace dict per test (last write wins), so those
scenarios can't be schema-asserted without multi-trace capture infra.

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from typing import Dict, List, Optional

from pydantic_ai import Agent

from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings
from deepeval.tracing import next_agent_span, next_llm_span


def create_next_span_agent(
    name: str = "pydanticai-next-span-test",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Agent:
    """A plain LLM-only agent. We deliberately do NOT bake
    ``metric_collection`` into settings so the staged LLM-span value
    has no trace-level peer to confuse precedence."""
    settings = DeepEvalInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "next-span"],
        metadata=metadata or {"test_type": "next_span"},
        thread_id=thread_id,
        user_id=user_id,
    )

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one short sentence only.",
        instrument=settings,
        name="next_span_agent",
    )


def invoke_with_next_llm_span(
    prompt: str,
    agent: Agent,
    llm_metric_collection: str,
    llm_metadata: Optional[Dict] = None,
) -> str:
    """``with next_llm_span(...)`` only — no agent-span staging.

    Asserts that LLM-span fields can be set independently of any other
    layer. The agent span should NOT carry ``metric_collection``.
    """
    with next_llm_span(
        metric_collection=llm_metric_collection,
        metadata=llm_metadata,
    ):
        return agent.run_sync(prompt).output


def invoke_with_stacked_next_spans(
    prompt: str,
    agent: Agent,
    agent_metric_collection: str,
    llm_metric_collection: str,
    agent_metadata: Optional[Dict] = None,
    llm_metadata: Optional[Dict] = None,
) -> str:
    """``with next_agent_span(...), next_llm_span(...)`` stacked.

    Asserts the typed slots are independent: the agent span gets the
    agent-staged values and the LLM span gets the LLM-staged values,
    no cross-talk. Mirrors scenario 2 of ``pydantic_after_next_span.py``.
    """
    with next_agent_span(
        metric_collection=agent_metric_collection,
        metadata=agent_metadata,
    ), next_llm_span(
        metric_collection=llm_metric_collection,
        metadata=llm_metadata,
    ):
        return agent.run_sync(prompt).output
