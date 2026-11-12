"""A reference agent for Voice Skills: one speech-to-speech model given the
journey's policy, records and tools, so models can be compared like for like."""

import asyncio
import json
import os
import re
from functools import partial
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List

from deepeval.voice.connectors import (
    BaseVoiceConnector,
    OpenAIRealtimeConnector,
    PhonicConnector,
)

INSTRUCTIONS = """You are the phone agent on the customer line of a company in this industry: {domain}. This is a live phone call.

Speak the way a good human agent does: short sentences, one question at a time, and read numbers, codes and names back clearly.
Use your tools to look things up and to make changes. Never invent account details, and never say something has been done unless the tool call for it succeeded.
If you did not catch something, ask the caller to repeat it instead of guessing.

How every call is handled:
- Find out what the caller needs, then identify their account, booking or order before discussing or changing it: ask for the reference or number and for the details that prove who they are, one at a time, then look it up.
- Ask, do not tell. Never read out personal details from a record so that the caller can agree to them.
- Read every code, number, date and name back to the caller and wait for a yes before you use it.
- If a lookup finds nothing, read back what you heard, character by character, and ask the caller to correct it.
- Before you make a change, say what you are about to do and get the caller's agreement. Afterwards, say what was done.
Today is {today}.
{sections}"""

_SETTINGS = ("today", "policy", "voice_style", "tools", "tool_order", "knowledge")
LANGUAGES = ["es", "fr", "pt", "pl"]
_READS = (
    "lookup", "look_up", "get", "find", "verify", "check", "search", "list",
    "track", "trace", "fetch", "retrieve", "view", "locate", "identify",
    "validate", "authenticate", "run", "quote", "read", "show", "test",
)


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


def _matches(arguments: dict, record: Dict) -> bool:
    """Whether a lookup's details find the record.

    References, numbers and codes have to be exact, the way a real system
    keys on them. A name only has to be close, and may be left out.
    """
    exact = 0
    for key, value in record.items():
        given = arguments.get(key)
        if given in (None, ""):
            continue
        if re.search(r"\d", str(value)):
            if _plain(given) != _plain(value):
                return False
            exact += 1
        elif (
            SequenceMatcher(None, _plain(given), _plain(value)).ratio() < 0.6
            and not any(re.search(r"\d", str(other)) for other in record.values())
        ):
            return False
    has_keys = any(re.search(r"\d", str(value)) for value in record.values())
    return exact > 0 or not has_keys


def _handler(name: str, mock: Dict, expected: List[Dict]) -> Callable:
    calls = {"made": 0}

    async def run(arguments: dict) -> Any:
        calls["made"] += 1
        await asyncio.sleep(mock.get("latency_seconds", 0))
        if calls["made"] == 1 and "first_call_returns" in mock:
            return mock["first_call_returns"]
        # A lookup with details that match nothing finds nothing.
        if name.startswith(_READS) and expected:
            if not any(_matches(arguments, call) for call in expected):
                return {
                    "error": "not_found",
                    "message": "No record matches those details.",
                }
        return mock.get("returns", {"status": "ok"})

    return run


def reference_agent(
    journey: Dict, model: str, dataset_path: str
) -> BaseVoiceConnector:
    """The reference agent for one journey, run on `model`.

    A model named `phonic-...` runs on Phonic; anything else on OpenAI's
    Realtime API.
    """
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
    connector = OpenAIRealtimeConnector
    if model.startswith("phonic"):
        # Phonic only speaks the languages it is told about, so it is given
        # every language a caller in the benchmark may switch to.
        connector = partial(PhonicConnector, additional_languages=LANGUAGES)
    return connector(
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
