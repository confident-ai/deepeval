import os
import json
from typing import Any, Dict
from openai import OpenAI
from deepeval.tracing.tracing import observe
from tests.test_integrations.utils import assert_trace_json, generate_trace_json


# 1) Define a local "tool" implementation (runs in your code)
@observe(type="tool")
def get_weather(location: str, unit: str = "c") -> Dict[str, Any]:
    # Demo stub: replace with a real API call if desired
    data = {
        "San Francisco": {"temp_c": 18, "condition": "partly cloudy"},
        "New York": {"temp_c": 22, "condition": "sunny"},
        "London": {"temp_c": 15, "condition": "light rain"},
    }
    city = location.strip()
    entry = data.get(city, {"temp_c": 20, "condition": "clear"})
    if unit.lower() == "f":
        temp = round(entry["temp_c"] * 9 / 5 + 32, 1)
        return {
            "location": city,
            "temperature": temp,
            "unit": "F",
            "condition": entry["condition"],
        }
    return {
        "location": city,
        "temperature": entry["temp_c"],
        "unit": "C",
        "condition": entry["condition"],
    }


# 2) Tool schema for Responses API (flatter format - name/parameters at top level)
TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["c", "f"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    }
]


@observe
def run_main():
    # Ensure your API key is set: export OPENAI_API_KEY=...
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = "You are a helpful assistant. Use tools when they are needed to get accurate data."
    user_prompt = (
        "What's the weather in San Francisco in celsius? "
        "Then give a one-sentence travel tip that fits the weather."
    )

    # 3) First call: model may request tool calls (Responses API)
    first = client.responses.create(
        model="gpt-4o-mini",
        instructions=system_prompt,
        input=user_prompt,  # simple text input is allowed
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )

    # Collect any function tool calls from output items
    tool_calls = []
    for item in first.output:
        if getattr(item, "type", None) == "function_call":
            # Fields: name, arguments (JSON str), call_id, id (optional)
            tool_calls.append(item)

    if tool_calls:
        # 4) Execute tools locally and send their outputs back using FunctionCallOutput items
        function_call_outputs = []
        for tc in tool_calls:
            name = tc.name
            args = json.loads(tc.arguments or "{}")

            if name == "get_weather":
                result = get_weather(**args)
            else:
                result = {"error": f"Unknown tool '{name}'"}

            function_call_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps(result),
                }
            )

        # 5) Second call: continue the same response thread with tool outputs
        final = client.responses.create(
            model="gpt-4o-mini",
            previous_response_id=first.id,
            input=function_call_outputs,
            temperature=0,
        )


_current_dir = os.path.dirname(os.path.abspath(__file__))


@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_tool_call_flow_response.json")
)
def test_tool_call_flow_response():
    run_main()
