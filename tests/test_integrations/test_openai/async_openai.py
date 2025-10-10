from deepeval.openai import AsyncOpenAI
from deepeval.tracing import trace
from deepeval.prompt import Prompt
import asyncio

client = AsyncOpenAI()

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")

async def test_async_openai_without_trace():
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )

async def test_async_openai_with_trace():
    with trace(
        prompt=prompt,
        thread_id="test_thread_id_1",
        llm_metric_collection="test_collection_1",
        name="test_name_1",
        tags=["test_tag_1"],
        metadata={"test_metadata_1": "test_value_1"},
        user_id="test_user_id_1",
    ):
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

async def test_async_response_create_without_trace():
    response = await client.responses.create(
        model="gpt-4o",
        instructions="You are a helpful assistant. Always generate a string response.",
        input="Hello, how are you?",
    )

async def test_async_response_create_with_trace():
    with trace(
        prompt=prompt,
        thread_id="test_thread_id_1",
        llm_metric_collection="test_collection_1",
        name="test_name_1",
        tags=["test_tag_1"],
        metadata={"test_metadata_1": "test_value_1"},
        user_id="test_user_id_1",
    ):
        response = await client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant. Always generate a string response.",
            input="Hello, how are you?",
        )

async def main():
    await test_async_openai_without_trace()
    await test_async_openai_with_trace()
    await test_async_response_create_without_trace()
    await test_async_response_create_with_trace()

if __name__ == "__main__":
    asyncio.run(main())
