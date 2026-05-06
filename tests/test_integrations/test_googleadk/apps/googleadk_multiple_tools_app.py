"""Multi-tool agent (weather + time).

Mirrors ``apps/agentcore_multiple_tools_app.py``. Drives both a
single-tool flow and a parallel-tools flow depending on the prompt.
The fixed mock data lets us assert specific substrings (``"72"`` /
``"sunny"`` / ``"7:00"`` / ``"GMT"`` / etc.) in the tests.
"""

from __future__ import annotations

import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from deepeval.integrations.google_adk import instrument_google_adk


_APP_NAME = "deepeval-googleadk-multiple-tools"


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name (e.g. ``"Tokyo"``).

    Returns:
        A dict with a ``report`` key holding the weather string.
    """
    weather_data = {
        "tokyo": "Sunny, 72F",
        "london": "Rainy, 55F",
        "paris": "Cloudy, 62F",
    }
    return {
        "report": weather_data.get(
            city.lower(), f"Weather data not available for {city}"
        )
    }


def get_time(city: str) -> dict:
    """Get the current time for a city.

    Args:
        city: The city name (e.g. ``"Tokyo"``).

    Returns:
        A dict with a ``time`` key holding the formatted time string.
    """
    time_data = {
        "tokyo": "3:00 PM JST",
        "london": "7:00 AM GMT",
        "paris": "8:00 AM CET",
    }
    return {
        "time": time_data.get(
            city.lower(), f"Time data not available for {city}"
        )
    }


def _build_agent() -> LlmAgent:
    return LlmAgent(
        model="gemini-2.0-flash",
        name="multi_tool_assistant",
        instruction=(
            "You have access to weather and time tools. "
            "When asked about weather, use get_weather. "
            "When asked about time, use get_time. Be concise."
        ),
        tools=[get_weather, get_time],
    )


def init_multiple_tools_googleadk(
    name: str = "googleadk-multiple-tools-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
):
    """Trace-level setup for the multiple-tools fixture. Per-tool /
    per-agent metric collections belong on ``with next_*_span(...)``
    blocks at the call site, not here."""
    instrument_google_adk(
        name=name,
        tags=tags or ["googleadk", "multiple-tools"],
        metadata=metadata or {"test_type": "multiple_tools"},
        thread_id=thread_id,
        user_id=user_id,
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


def invoke_multiple_tools_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_multiple_tools_googleadk()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_multiple_tools_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_multiple_tools_googleadk()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
