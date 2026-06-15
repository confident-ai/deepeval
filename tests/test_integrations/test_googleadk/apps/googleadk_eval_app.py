"""Google ADK evals fixture — trace-level setup with an ADK tool that
mutates its own span via ``update_current_span``.

After the OTel POC migration, ``init_evals_googleadk(...)`` carries
ONLY trace-level kwargs. Per-call agent / LLM / tool metric collections
and ``BaseMetric`` instances are staged at the call site:

    with next_agent_span(metric_collection="agent_v1", metrics=[...]):
        with next_llm_span(metric_collection="llm_v1"):
            invoke_evals_agent(prompt, invoke_func=invoke_func)

The ADK tool ``special_tool`` uses ``update_current_span`` from inside
its body to set its own ``metric_collection`` — exercising the
placeholder push/pop path that flips Google ADK from "Bad" to "Good"
in the integrations matrix.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from deepeval.integrations.google_adk import instrument_google_adk
from deepeval.tracing import update_current_span


_APP_NAME = "deepeval-googleadk-evals"


def special_tool(query: str) -> dict:
    """A tool used by feature tests.

    Mutates its own span via ``update_current_span(...)`` so the
    placeholder push/pop pattern is exercised end-to-end. With the
    POC migration this lands on ``confident.span.metric_collection``
    of THIS tool span (no longer a no-op as it was under the old
    ``is_test_mode`` path).

    Args:
        query: The query string to process.

    Returns:
        A dict with a ``processed`` key holding the formatted result.
    """
    update_current_span(metric_collection="special_tool_v1")
    return {"processed": f"Processed: {query}"}


def _build_agent() -> LlmAgent:
    return LlmAgent(
        model="gemini-2.0-flash",
        name="evals_assistant",
        instruction="You are a helpful assistant. Be concise.",
        tools=[special_tool],
    )


def init_evals_googleadk(
    name: str = "googleadk-evals-test",
    tags: List[str] = None,
    metadata: Dict = None,
    thread_id: str = None,
    user_id: str = None,
    metric_collection: Optional[str] = None,
):
    """Wire deepeval OTel pipeline + an ADK agent with one
    ``update_current_span``-using tool. Trace-only kwargs."""
    instrument_google_adk(
        name=name,
        tags=tags or ["googleadk", "evals"],
        metadata=metadata or {"test_type": "evals"},
        thread_id=thread_id,
        user_id=user_id,
        metric_collection=metric_collection,
    )

    agent = _build_agent()
    runner = InMemoryRunner(agent=agent, app_name=_APP_NAME)

    async def _ainvoke(payload: dict) -> dict:
        prompt = payload.get("prompt", "")
        actor = payload.get("user_id") or "test-user"
        session = await runner.session_service.create_session(
            app_name=_APP_NAME, user_id=actor
        )
        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        text_output = ""
        async for event in runner.run_async(
            user_id=actor,
            session_id=session.id,
            new_message=content,
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts or []:
                    if getattr(part, "text", None):
                        text_output += part.text
        return {"result": text_output}

    def invoke(payload: dict) -> dict:
        return asyncio.run(_ainvoke(payload))

    invoke.ainvoke = _ainvoke
    return invoke


def invoke_evals_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_evals_googleadk()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_evals_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_evals_googleadk()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
