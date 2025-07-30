from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import (
    observe,
    update_current_span,
    LlmAttributes,
)
import asyncio
from openai import AsyncClient


async_client = AsyncClient()


@observe(
    type="llm",
    model="gpt-4o",
    cost_per_input_token=0.0000003,
    cost_per_output_token=0.0000025,
)
async def meta_agent(query: str):
    response = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
    )
    update_current_span(
        metadata={"user_id": "11111", "date": "1/1/11"},
        test_case=LLMTestCase(
            input="What is this again?",
            actual_output="this is a latte",
            expected_output="this is a mocha",
            retrieval_context=["I love coffee"],
            context=["I love coffee"],
            tools_called=[ToolCall(name="test")],
            expected_tools=[ToolCall(name="test")],
        ),
        attributes=LlmAttributes(
            input=query,
            output=response,
            input_token_count=response.usage.total_tokens,
            output_token_count=response.usage.total_tokens,
        ),
    )
    return (response.choices[0].message.content,)


async def run_parallel_examples():
    tasks = [
        meta_agent("How tall is Mount Everest?"),
        meta_agent("What's the capital of Brazil?"),
    ]
    await asyncio.gather(*tasks)


asyncio.run(run_parallel_examples())
