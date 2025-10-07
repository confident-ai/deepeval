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
    instrument=confident_instrumentation_settings,
    name="test_agent",
)


async def execute_simple_agent():
    with trace() as current_trace:
        result_1 = await agent.run("What are the LLMs?")
        print("===============Result 1: trace ID===============")
        print(current_trace.uuid) # ok

        result_2 = await agent.run("What are the LLMs?")
        print("===============Result 2: Trace ID:===============")
        print(current_trace.uuid) # ok

def execute_simple_agent_sync():

    with trace() as current_trace:
        result_1 = agent.run_sync("What are the LLMs?")
        print("===============Result 1: trace ID===============")
        print(current_trace.uuid) # ok

# Test nested trace() calls
def test_nested_traces():
    with trace() as outer_trace:
        outer_uuid = outer_trace.uuid
        print(f"Outer trace UUID: {outer_uuid}") # ok
        
        result_1 = agent.run_sync("Query 1")
        print(f"After agent run 1: {outer_trace.uuid}") # ok
        
        # Nested trace context
        with trace() as inner_trace:
            # Should reuse the same trace (based on trace_context.py logic)
            print(f"Inner trace UUID: {inner_trace.uuid}") # ok
            
            result_2 = agent.run_sync("Query 2")
            print(f"After agent run 2: {inner_trace.uuid}") # ok

# Test concurrent agent runs
async def test_concurrent_agents():
    with trace() as current_trace:
        initial_uuid = current_trace.uuid
        
        # Run multiple agents concurrently
        results = await asyncio.gather(
            agent.run("Query 1"),
            agent.run("Query 2"),
            agent.run("Query 3"),
        )
        
        print(f"Initial: {initial_uuid}")
        print(f"Final: {current_trace.uuid}") # print trace uuid of the last agent run
