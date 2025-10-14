from deepeval.openai import OpenAI, AsyncOpenAI
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.tracing import observe


RESPONSE_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia",
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    }
]

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


sync_client = OpenAI()
async_client = AsyncOpenAI()

from deepeval.tracing import trace

@observe()
def llm_app(
    input: str,
    completion_mode: str = "chat",
):
    if completion_mode == "chat":

        with trace(llm_metrics=[AnswerRelevancyMetric(), BiasMetric()]):
            response = sync_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful chatbot. Always generate a string response.",
                    },
                    {"role": "user", "content": input},
                ],
                # tools=CHAT_TOOLS,
            )
            return response.choices[0].message.content
    else:
        with trace(llm_metrics=[AnswerRelevancyMetric(), BiasMetric()]):
            response = sync_client.responses.create(
                model="gpt-4o",
                instructions="You are a helpful assistant. Always generate a string response.",
                input=input,
                # tools=RESPONSE_TOOLS,
            )
            return response.output_text


@observe()
async def async_llm_app(
    input: str,
    completion_mode: str = "chat",
):
    if completion_mode == "chat":
        with trace(llm_metrics=[AnswerRelevancyMetric(), BiasMetric()]):
            response = await async_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful chatbot. Always generate a string response.",
                    },
                    {"role": "user", "content": input},
                ],
                # tools=CHAT_TOOLS,
            )
        return response.choices[0].message.content
    else:
        with trace(llm_metrics=[AnswerRelevancyMetric(), BiasMetric()]):
            response = await async_client.responses.create(
                model="gpt-4o",
                instructions="You are a helpful assistant. Always generate a string response.",
                input=input,
                # tools=RESPONSE_TOOLS,
            )
        return response.output_text
