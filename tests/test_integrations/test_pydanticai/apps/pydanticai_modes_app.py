"""PydanticAI Modes App: validates the three execution modes documented in
``deepeval/integrations/pydantic_ai/README.md``.

  - Mode 1 — bare ``agent.run_sync(...)`` with `update_current_trace` /
    `update_current_span` from inside a tool body. Implicit ``Trace``
    placeholder pushed by ``SpanInterceptor.on_start`` is the write target.
    Mirrors ``pydantic_after_bare.py``.
  - Mode 2 — ``with trace(...)`` wrapper. User-pushed ``Trace`` (non-implicit),
    so routing flips to REST and the deepeval-managed trace owns the lifecycle.
  - Mode 3 — ``@observe`` decorator. Symmetric to Mode 2 from this integration's
    perspective; adds an outer deepeval span around the agent call. Mirrors
    ``pydantic_after.py``.

Uses deterministic settings for reproducible traces. The tool body in the
enrichment variant deliberately writes to BOTH the trace (via
``update_current_trace``) and the tool span (via ``update_current_span``)
so a single trace exercises both write targets.
"""

from typing import Dict, List, Optional

from pydantic_ai import Agent

from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings
from deepeval.tracing import (
    observe,
    trace,
    update_current_span,
    update_current_trace,
)


def create_modes_agent(
    name: str = "pydanticai-modes-test",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Agent:
    """A plain LLM-only agent for `@observe` / `with trace(...)` tests.

    No tools — these tests only need to validate the trace-shape under
    each mode's routing path, not tool behavior.
    """
    settings = DeepEvalInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "modes"],
        metadata=metadata or {"test_type": "modes"},
        thread_id=thread_id,
        user_id=user_id,
    )

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt="Be concise, reply with one short sentence only.",
        instrument=settings,
        name="modes_agent",
    )


def invoke_in_observe_mode(
    prompt: str,
    agent: Agent,
    outer_name: str = "observe-outer",
    trace_name: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """Run the agent inside ``@observe(type="agent")``.

    Mirrors ``pydantic_after.py``. The outer ``@observe``-decorated
    function pushes a non-implicit ``Trace`` onto ``current_trace_context``,
    so routing flips to REST. ``update_current_trace(...)`` from inside
    the body lands on the user-pushed trace (not the implicit
    placeholder, which isn't pushed because there's already a real one).

    Returns the agent's output. The outer span will appear as the
    deepeval-managed agent-type root in the resulting trace JSON,
    with pydantic-ai's own agent/llm spans nested underneath.
    """

    @observe(type="agent", name=outer_name)
    def _outer(p: str) -> str:
        update_current_trace(
            name=trace_name,
            user_id=user_id,
            tags=tags,
            metadata=metadata,
        )
        return agent.run_sync(p).output

    return _outer(prompt)


def invoke_in_with_trace_mode(
    prompt: str,
    agent: Agent,
    trace_name: str,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """Run the agent inside ``with trace(...)``.

    Same routing outcome as ``@observe`` (REST), but no outer
    deepeval-managed span — the trace tree is just pydantic-ai's
    own agent/llm spans under the user-pushed ``Trace``.
    """
    with trace(
        name=trace_name,
        user_id=user_id,
        thread_id=thread_id,
        tags=tags,
        metadata=metadata,
    ):
        return agent.run_sync(prompt).output


def create_enrichment_agent(
    name: str = "pydanticai-enrichment-test",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Agent:
    """Agent whose ``lookup`` tool enriches BOTH the trace and the tool
    span via ``update_current_trace`` and ``update_current_span``.

    Used in bare mode (no ``@observe`` / ``with trace(...)``) to prove
    the implicit ``Trace`` placeholder push works end-to-end:
    ``update_current_trace`` from inside a tool mutates the implicit
    placeholder, the value is serialized at every span's ``on_end``
    into ``confident.trace.*`` OTel attrs, and the captured trace JSON
    reflects it.
    """
    settings = DeepEvalInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "enrichment"],
        metadata=metadata or {"test_type": "bare_tool_enrichment"},
        thread_id=thread_id,
        user_id=user_id,
    )

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt=(
            "You are an assistant. Use the lookup tool whenever the user "
            "mentions a key. Be concise."
        ),
        instrument=settings,
        name="enrichment_agent",
    )

    @agent.tool_plain
    def lookup(key: str) -> str:
        """Look up a value for a key. Enriches the active trace AND the
        tool span with derived metadata."""
        update_current_span(
            metadata={
                "tool_called": True,
                "lookup_key": key,
            },
        )
        update_current_trace(
            metadata={
                "enriched_from_tool": True,
                "resolved_key": key,
            },
        )
        return f"resolved-value-for-{key}"

    return agent


def invoke_with_tool_enrichment(
    prompt: str,
    agent: Agent,
) -> str:
    """Bare ``agent.run_sync`` — no ``@observe`` / ``with trace(...)``.
    The implicit ``Trace`` placeholder is pushed by ``SpanInterceptor``
    when the OTel root span starts; the tool body's
    ``update_current_trace(...)`` mutates it.
    """
    return agent.run_sync(prompt).output
