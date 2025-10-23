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


# 2) Tool schema exposed to the model
TOOLS = [
    {
        "type": "function",
        "function": {
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 3) First call: model may request tool calls
    first = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )

    assistant_msg = first.choices[0].message
    tool_calls = assistant_msg.tool_calls or []

    # If the model called tools, run them locally and provide results back
    if tool_calls:
        # Add the assistant message that asked for tools
        messages.append(
            {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    tc.model_dump() for tc in tool_calls
                ],  # keep structure for continuity
            }
        )

        # Execute each tool and send its result
        for tc in tool_calls:
            if tc.type == "function":
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                if name == "get_weather":
                    result = get_weather(**args)
                else:
                    result = {"error": f"Unknown tool '{name}'"}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )

        # 4) Second call: model composes a final answer using tool outputs
        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )


_current_dir = os.path.dirname(os.path.abspath(__file__))


@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_tool_call_flow_completion.json")
)
def test_tool_call_flow_completion():
    run_main()
