"""A reference agent for Voice Skills: one speech-to-speech model given the
journey's policy, records and tools, so models can be compared like for like."""

import asyncio
import json
import os
import re
from typing import Any, Callable, Dict, List

from deepeval.voice.connectors import OpenAIRealtimeConnector

INSTRUCTIONS = """You are the phone agent on the customer line of a company in this industry: {domain}. This is a live phone call.

Speak the way a good human agent does: short sentences, one question at a time, and read numbers, codes and names back clearly.
Use your tools to look things up and to make changes. Never invent account details, and never say something has been done unless the tool call for it succeeded.
If you did not catch something, ask the caller to repeat it instead of guessing.
Today is {today}.
{sections}"""

_SETTINGS = ("today", "policy", "voice_style", "tools", "tool_order", "knowledge")
_READS = ("lookup", "get", "find", "verify", "check", "search", "list")


def _plain(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _sections(setup: Dict) -> str:
    parts = []
    if setup.get("policy"):
        parts.append(f'Company policy, which you must follow:\n{setup["policy"]}')
    if setup.get("voice_style"):
        parts.append(f'How you should sound:\n{setup["voice_style"]}')
    if setup.get("knowledge"):
        parts.append(
            "What you know:\n"
            + json.dumps(setup["knowledge"], indent=1, default=str)
        )
    return "\n\n".join(parts)


def _records(setup: Dict) -> Dict:
    """The customer's records, which the agent only sees by looking them up."""
    return {
        key: value
        for key, value in setup.items()
        if key not in _SETTINGS
    }


def _tool_definitions(journey: Dict, results: Dict) -> List[dict]:
    parameters: Dict[str, Dict[str, dict]] = {}
    for call in journey["expected_tools"] + journey["forbidden_tools"]:
        known = parameters.setdefault(call["name"], {})
        for name in call["input_parameters"]:
            known[name] = {"type": "string"}
    for name in results:
        parameters.setdefault(name, {})
    return [
        {
            "type": "function",
            "name": name,
            "description": name.replace("_", " ").capitalize() + ".",
            "parameters": {"type": "object", "properties": properties},
        }
        for name, properties in parameters.items()
    ]


def _handler(name: str, mock: Dict, expected: List[Dict]) -> Callable:
    calls = {"made": 0}

    async def run(arguments: dict) -> Any:
        calls["made"] += 1
        await asyncio.sleep(mock.get("latency_seconds", 0))
        if calls["made"] == 1 and "first_call_returns" in mock:
            return mock["first_call_returns"]
        # A lookup with details that match nothing finds nothing.
        if name.startswith(_READS) and expected:
            found = any(
                all(
                    _plain(arguments.get(key)) == _plain(value)
                    for key, value in call.items()
                )
                for call in expected
            )
            if not found:
                return {
                    "error": "not_found",
                    "message": "No record matches those details.",
                }
        return mock.get("returns", {"status": "ok"})

    return run


def reference_agent(
    journey: Dict, model: str, dataset_path: str
) -> OpenAIRealtimeConnector:
    """The reference agent for one journey, run on `model`."""
    setup = journey.get("agent_setup") or {}
    results_path = os.path.join(dataset_path, "tool_results.json")
    results: Dict[str, Dict] = {}
    if os.path.exists(results_path):
        with open(results_path) as file:
            results = {
                name: {"returns": returns}
                for name, returns in json.load(file)
                .get(journey["id"], {})
                .items()
            }
    results.update(setup.get("tools") or {})
    records = _records(setup)
    lookup = next(
        (
            call["name"]
            for call in journey["expected_tools"]
            if call["name"].startswith(_READS)
        ),
        None,
    )
    if records and lookup:
        found = dict(results.get(lookup, {}))
        found["returns"] = {**records, **(found.get("returns") or {})}
        results[lookup] = found
    tools = _tool_definitions(journey, results)
    expected: Dict[str, List[Dict]] = {}
    for call in journey["expected_tools"]:
        expected.setdefault(call["name"], []).append(call["input_parameters"])
    return OpenAIRealtimeConnector(
        model=model,
        instructions=INSTRUCTIONS.format(
            domain=journey["domain"],
            today=setup.get("today", ""),
            sections=_sections(setup),
        ),
        tools=tools,
        tool_handlers={
            tool["name"]: _handler(
                tool["name"],
                results.get(tool["name"], {}),
                expected.get(tool["name"], []),
            )
            for tool in tools
        },
        turn_detection="patient",
    )
