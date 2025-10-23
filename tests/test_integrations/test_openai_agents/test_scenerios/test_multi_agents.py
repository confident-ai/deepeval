import os
import pytest
import asyncio
import json

from agents import Agent, Runner, add_trace_processor
from deepeval.openai_agents.callback_handler import DeepEvalTracingProcessor
from tests.test_integrations.utils import (
    assert_json_object_structure,
    load_trace_data,
)
from deepeval.tracing.trace_test_manager import trace_testing_manager

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


################################ TESTING CODE #################################

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "multi_agents.json")


async def test_json_schema():
    """
    Test the json schema of the trace. Raises an exception if the schema is invalid.
    """
    try:
        trace_testing_manager.test_name = json_path
        await run()
        actual_dict = await trace_testing_manager.wait_for_test_dict()
        expected_dict = load_trace_data(json_path)

        assert assert_json_object_structure(expected_dict, actual_dict)
    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None


################################ Generate Actual JSON Dump Code #################################


async def generate_actual_json_dump():
    try:
        trace_testing_manager.test_name = json_path
        await run()
        actual_dict = await trace_testing_manager.wait_for_test_dict()

        with open(json_path, "w") as f:
            json.dump(actual_dict, f)
    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None
