import asyncio
from deepeval.tracing import trace
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)
from deepeval.prompt import Prompt


prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")


confident_instrumentation_settings = ConfidentInstrumentationSettings(
    thread_id="test_thread_id_1",
    user_id="test_user_id_1",
    metadata={"streaming": True, "prompt_version": "00.00.11"},
    tags=["test_tag_1", "test_tag_2"],
    metric_collection="test_metric_collection_1",
    name="test_name_1",
    confident_prompt=prompt,
    trace_metric_collection="test_collection_1",
)

agent = Agent(
    "openai:gpt-5",
    system_prompt="Be concise, reply with one sentence.",
    instructions="You are a helpful assistant.",
    instrument=confident_instrumentation_settings,
    name="test_agent",
)


async def execute_simple_agent():
    with trace():
        await agent.run("What are the LLMs?")
        await agent.run("What are the LLMs?")


def execute_simple_agent_sync():

    with trace():
        agent.run_sync("What are the LLMs?")


#################### Testing different trace modes #################################


def nested_traces():
    with trace():
        agent.run_sync("Query 1")

        # Nested trace context
        with trace():
            agent.run_sync("Query 2")


# Test concurrent agent runs
async def concurrent_agents():
    with trace():

        await asyncio.gather(
            agent.run("Query 1"),
            agent.run("Query 2"),
            agent.run("Query 3"),
        )
