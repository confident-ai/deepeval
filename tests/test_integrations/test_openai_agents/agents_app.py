import asyncio
from agents import Runner, set_trace_processors
from deepeval.tracing import update_current_trace
from agents import trace
from multi_agents import triage_agent
from weather_agent import weather_agent
from weather_agent_patched import weather_agent_patched

from deepeval.openai_agents.callback_handler import DeepEvalTracingProcessor

set_trace_processors([DeepEvalTracingProcessor()])

async def main1():
    # with trace (group_id and metadata)
    with trace(
        workflow_name="test_workflow_1", # name of the trace,
        group_id="test_group_id_1", # thread_id of the trace,
        metadata={"test_metadata_1": "test_metadata_1"}, # metadata of the trace,
    ):
        user_query = "What's the weather like in London today?"
        response_1 = await Runner.run( # since thread id is present, input should not contain unwanted information.
            triage_agent,
            "Hola, ¿cómo estás?",
        )
        response_2 = await Runner.run(
            weather_agent, user_query
        )
        # the input is of the first run and the output is of the second run.


# without trace (group_id and metadata not present)
async def main2():
    user_query = "What's the weather like in London today?"
    response_1 = await Runner.run(
        triage_agent,
        "Hola, ¿cómo estás?",
    ) # cleaned output
    response_2 = await Runner.run(
        weather_agent, user_query
    ) # ouput will contain some unwanted information.


async def main3():
    user_query = "What's the weather like in London today?"
    with trace(
        workflow_name="test_workflow_1",
        group_id="test_group_id_1",
        metadata={"test_metadata_1": "test_metadata_1"},
    ):
        response_2 = await Runner.run(weather_agent, user_query)
        update_current_trace(input="initial input", output="final output")
    
    with trace(
        workflow_name="test_workflow_2",
        group_id="test_group_id_2",
        metadata={"test_metadata_2": "test_metadata_2"},
    ):
        response_1 = await Runner.run(
            triage_agent,
            "Hola, ¿cómo estás?",
        )


async def main4():
    user_query = "What's the weather like in London today?"
    with trace(
        workflow_name="test_workflow_1",
        group_id="test_group_id_1",
        metadata={"test_metadata_1": "test_metadata_1"},
    ):
        run_streamed_1 = Runner.run_streamed(
            weather_agent, user_query
        )
        async for chunk in run_streamed_1.stream_events():
            continue
        
        run_streamed_2 = Runner.run_streamed(
            triage_agent,
            "Hola, ¿cómo estás?",
        )
        async for chunk in run_streamed_2.stream_events():
            continue

async def main5():
    await Runner.run(
        weather_agent_patched,
        "What's the weather in London?",
    )


async def execute_agent():
    # await main1()
    # await main2()
    # await main3()
    # await main4()
    await main5()

if __name__ == "__main__":
    asyncio.run(execute_agent())