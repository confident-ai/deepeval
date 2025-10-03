import os
import pytest
import asyncio

from agents import Agent, Runner, add_trace_processor
from deepeval.openai_agents.callback_handler import DeepEvalTracingProcessor
from deepeval.tracing.utils import assert_json_file_structure

add_trace_processor(DeepEvalTracingProcessor())

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)

async def run():
    await Runner.run(triage_agent, "Hola, ¿cómo estás?")


def generate_actual_json_dump():
    try:
        actual_path = '../trace_dump/run_multi_agents.json'
        original_value = os.environ.get('DEEPEVAL_TRACING_TEST_PATH')
        os.environ['DEEPEVAL_TRACING_TEST_PATH'] = actual_path
        asyncio.run(run())
    finally:
        if original_value is not None:
            os.environ['DEEPEVAL_TRACING_TEST_PATH'] = original_value
        else:
            os.environ.pop('DEEPEVAL_TRACING_TEST_PATH', None)

@pytest.mark.asyncio
async def test_json_schema():
    """
    Test the json schema of the trace. Raises an exception if the schema is invalid.
    """
    expected_temp_path = '../trace_dump/temp_run_multi_agents.json'
    actual_temp_path = '../trace_dump/run_multi_agents.json'
    
    original_value = os.environ.get('DEEPEVAL_TRACING_TEST_PATH')
    
    try:
        os.environ['DEEPEVAL_TRACING_TEST_PATH'] = expected_temp_path
        # This will raise an exception if there are any schema validation errors
        await run()
        assert assert_json_file_structure(expected_temp_path, actual_temp_path)
    
    finally:
        if original_value is not None:
            os.environ['DEEPEVAL_TRACING_TEST_PATH'] = original_value
        else:
            os.environ.pop('DEEPEVAL_TRACING_TEST_PATH', None)
        
        # Delete the expected temp file
        if os.path.exists(expected_temp_path):
            os.remove(expected_temp_path)


# generate_actual_json_dump()