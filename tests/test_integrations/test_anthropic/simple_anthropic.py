from anthropic import Anthropic, AsyncAnthropic
from deepeval.tracing import LlmSpanContext, trace, observe
from deepeval.prompt import Prompt

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"

client = Anthropic()
async_client = AsyncAnthropic()

with trace(
    llm_span_context=LlmSpanContext(
        prompt=prompt,
        metric_collection="test_collection_1",
    ),
    thread_id="test_thread_id_1",
):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        system="You are a helpful assistant.",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )


@observe()
async def run_async_anthropic():
    with trace(llm_span_context=LlmSpanContext(prompt=prompt)):
        await async_client.messages.create(
            model="claude-sonnet-4-5",
            system="You are a helpful assistant.",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "Hello, how are you?"},
            ],
        )
