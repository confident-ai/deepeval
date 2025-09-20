from deepeval.prompt import Prompt
from deepeval.integrations.pydantic_ai import Agent
from deepeval.tracing import observe
import asyncio


@observe(type="tool", metric_collection="test_collection_1")
def get_weather(city: str) -> str:
    """Gets the weather for a given city."""
    return f"I don't know the weather for {city}."


prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")

agent = Agent(
    "openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful weather agent.",
    trace_name="test_name_1",
    trace_tags=["test_tag_1"],
    trace_metadata={"test_metadata_1": "test_metadata_1"},
    trace_thread_id="test_thread_id_1",
    trace_user_id="test_user_id_1",
    trace_metric_collection="test_collection_1",
    llm_metric_collection="test_collection_1",
    llm_prompt=prompt,
    agent_metric_collection="test_collection_1",
)


async def execute_agent_stream():
    async with agent.run_stream(
        "What is the weather in London?", name="test_name_2"
    ) as result:
        async for chunk in result.stream_text(delta=True):
            print(chunk, end="", flush=True)
        final = await result.get_output()
        print("\n\nFinal:", final)


async def execute_agent_run():
    result = await agent.run(
        "What is the weather in London?", name="test_name_4"
    )
    print(result.output)


def execute_all():
    asyncio.run(execute_agent_stream())
    agent.run_sync("What is the weather in London?", name="test_name_3")
    asyncio.run(execute_agent_run())
