from openai import OpenAI, AsyncOpenAI
from deepeval.tracing import trace, observe, LlmSpanContext
from deepeval.prompt import Prompt


prompt = Prompt(alias="asd")
prompt._version = "00.00.01"

client = OpenAI()
async_client = AsyncOpenAI()

with trace(
    llm_span_context=LlmSpanContext(
        prompt=prompt,
        metric_collection="test_collection_1",
    ),
    thread_id="test_thread_id_1",
):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },  # String system prompt
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )


@observe()
async def run_async_openai():
    with trace(llm_span_context=LlmSpanContext(prompt=prompt)):
        await async_client.responses.create(
            model="gpt-4o-mini",
            instructions="You are a helpful assistant.",
            input="Hello, how are you?",
        )
