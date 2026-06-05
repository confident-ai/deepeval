"""LangChain Next-Span App: validates ``with next_llm_span(...)`` against
a real ``ChatOpenAI`` driving ``create_agent``.

Mirrors the pydantic_ai ``pydanticai_next_span_app.py`` pattern: closes
the schema-test coverage gap for ``next_llm_span`` by exercising the
``CallbackHandler``'s ``pop_pending_for("llm")`` +
``apply_pending_to_span(...)`` plumbing through a real LLM trace shape
(token counts, response_metadata, etc.) — not just in-memory span
attributes the way the unit tests in ``test_next_span.py`` do.

We deliberately do NOT bake ``metric_collection`` into the
``ChatOpenAI(metadata=...)`` baseline so the staged LLM-span value has
no metadata-level peer that could confuse the precedence story
(``next_llm_span`` always wins on overlap — see comment in
``deepeval/integrations/langchain/callback.py::on_llm_start``).

The "one-shot" semantic is a deliberate part of the schema shape: the
agent loop's SECOND LLM span (after the tool result is fed back in)
must NOT carry ``metric_collection`` — only the first one does.
"""

from typing import Dict, Optional

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from deepeval.tracing import next_llm_span


@tool
def square(n: int) -> int:
    """Returns the square of the input integer."""
    return n * n


_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0, seed=42)

_agent_executor = create_agent(
    _llm,
    [square],
    system_prompt=(
        "You are a math assistant. Always call the `square` tool to compute "
        "squares; do not compute them yourself. After the tool result, reply "
        "with the integer result and nothing else."
    ),
)


def invoke_with_next_llm_span(
    inputs: dict,
    metric_collection: str,
    metadata: Optional[Dict] = None,
    config: RunnableConfig = None,
):
    """Wrap the agent invocation in ``with next_llm_span(...)``.

    Stages ``metric_collection`` (and optional ``metadata``) onto the
    NEXT LLM span the callback opens — which is the agent loop's first
    chat-model call. The second chat-model call (after the tool
    response is appended) sees an empty pending slot and ends up with
    ``metric_collection=None`` in the trace.
    """
    with next_llm_span(
        metric_collection=metric_collection,
        metadata=metadata,
    ):
        return _agent_executor.invoke(inputs, config=config)


async def ainvoke_with_next_llm_span(
    inputs: dict,
    metric_collection: str,
    metadata: Optional[Dict] = None,
    config: RunnableConfig = None,
):
    """Async counterpart of ``invoke_with_next_llm_span``. The pending
    slot uses ``ContextVar`` semantics so ``await`` boundaries inside
    the agent's chat-model call do not drop the staged value before
    ``on_chat_model_start`` pops it."""
    with next_llm_span(
        metric_collection=metric_collection,
        metadata=metadata,
    ):
        return await _agent_executor.ainvoke(inputs, config=config)
