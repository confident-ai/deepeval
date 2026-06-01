"""Single-tool calculator agent.

Mirrors ``apps/agentcore_tool_app.py``. The ``calculate`` tool is a
plain Python function with type hints + docstring — Google ADK
auto-wraps it into a FunctionTool when the agent is constructed.
"""

from __future__ import annotations

import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from deepeval.integrations.google_adk import instrument_google_adk


_APP_NAME = "deepeval-googleadk-tool"


def calculate(operation: str, a: float, b: float) -> dict:
    """Perform basic arithmetic operations.

    Args:
        operation: One of ``add``, ``subtract``, ``multiply``, ``divide``.
        a: The first operand.
        b: The second operand.

    Returns:
        A dict with a ``result`` key holding the numeric result.
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }
    op_func = operations.get(operation.lower())
    if op_func is None:
        return {"error": f"Unsupported operation: {operation}"}
    return {"result": op_func(a, b)}


def _build_agent() -> LlmAgent:
    return LlmAgent(
        model="gemini-2.0-flash",
        name="calculator_assistant",
        instruction=(
            "You are a calculator assistant. Use the calculate tool for "
            "math operations. Be concise."
        ),
        tools=[calculate],
    )


def init_tool_googleadk(
    name: str = "googleadk-tool-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
):
    """Trace-only setup. Tool / agent / LLM span-level fields belong at
    the call site (``with next_*_span(...)`` or ``update_current_span``
    inside the tool body)."""
    instrument_google_adk(
        name=name,
        tags=tags or ["googleadk", "tool"],
        metadata=metadata or {"test_type": "tool"},
        thread_id=thread_id,
        user_id=user_id,
    )

    agent = _build_agent()
    runner = InMemoryRunner(agent=agent, app_name=_APP_NAME)

    async def _ainvoke(payload: dict) -> dict:
        prompt = payload.get("prompt", "What is 7 multiplied by 8?")
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


def invoke_tool_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_tool_googleadk()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_tool_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_tool_googleadk()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
