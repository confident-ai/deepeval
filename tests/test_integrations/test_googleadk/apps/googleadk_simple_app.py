"""Simple greeting agent — no tools, just an LLM call.

Mirrors ``apps/agentcore_simple_app.py``. Trace-only kwargs at the
``init_simple_googleadk(...)`` boundary; span-level config goes on
``with next_*_span(...)`` / ``update_current_span(...)`` at the call site.
"""

from __future__ import annotations

import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from deepeval.integrations.google_adk import instrument_google_adk


_APP_NAME = "deepeval-googleadk-simple"


def _build_agent() -> LlmAgent:
    return LlmAgent(
        model="gemini-2.0-flash",
        name="simple_assistant",
        instruction=(
            "You are a concise assistant. Reply with one short sentence only."
        ),
    )


def init_simple_googleadk(
    name: str = "googleadk-simple-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
):
    """Wire the deepeval OTel pipeline and build an ADK agent.

    All kwargs are trace-level. Span-level configuration belongs at the
    call site via ``with next_*_span(...)`` blocks or
    ``update_current_span(...)`` from inside an ADK tool body.
    """
    instrument_google_adk(
        name=name,
        tags=tags or ["googleadk", "simple"],
        metadata=metadata or {"test_type": "simple"},
        thread_id=thread_id,
        user_id=user_id,
    )

    agent = _build_agent()
    runner = InMemoryRunner(agent=agent, app_name=_APP_NAME)

    async def _ainvoke(payload: dict) -> dict:
        prompt = payload.get("prompt", "Hello!")
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


def invoke_simple_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_simple_googleadk()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_simple_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_simple_googleadk()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
