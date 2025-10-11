import asyncio
from deepeval.openai import OpenAI
from deepeval.tracing import trace, observe
from deepeval.prompt import Prompt

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")

from deepeval.openai import AsyncOpenAI

client = OpenAI()
async_client = AsyncOpenAI()

with trace(
    prompt=prompt,
    thread_id="test_thread_id_1",
    llm_metric_collection="test_collection_1",
):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."}, # String system prompt
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )

@observe()
async def run_async_openai():
    with trace(prompt=prompt) as current_trace:
        response = await async_client.responses.create(
            model="gpt-4o-mini",
            instructions="You are a helpful assistant.",
            input="Hello, how are you?",
        )
        print(current_trace.uuid)
