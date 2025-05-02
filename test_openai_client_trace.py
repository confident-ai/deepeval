import time

from openai import OpenAI

import deepeval
from deepeval.tracing import observe

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
        },
    }
]


deepeval.login_with_confident_api_key("<your-deepeval-api-key>")
# Initialize OpenAI client
client = OpenAI(api_key="<your-openai-api-key>")


@observe(type="llm", llm_provider="openai", client=client)
def generate_response(input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or your preferred model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ],
        temperature=0.7,
        # tools=tools,
        # tool_choice="auto"
    )

    # response = client.beta.chat.completions.parse(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": input}
    #     ],
    # )
    return response


try:
    response = generate_response("What is the weather in Tokyo?")
    print(response)
except Exception as e:
    raise e

time.sleep(5)
