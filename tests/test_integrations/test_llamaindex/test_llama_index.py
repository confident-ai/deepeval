import os
import json

import pytest
from tests.test_integrations.utils import (
    assert_json_object_structure,
    load_trace_data,
)
from deepeval.tracing.trace_test_manager import trace_testing_manager
from llama_index.llms.openai import OpenAI

from deepeval.integrations.llama_index import (
    FunctionAgent,
)


def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can perform calculations.",
    metric_collection="test_collection_1",
)


async def llm_app(input: str):
    return await agent.run(input)


################################ TESTING CODE #################################

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "llama_index.json")


@pytest.mark.skip(reason="LlamaIndex is deprecated")
async def test_json_schema():
    """
    Test the json schema of the trace. Raises an exception if the schema is invalid.
    """
    try:
        trace_testing_manager.test_name = json_path
        await llm_app("What is 3 * 12?")
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
        await llm_app("What is 3 * 12?")
        actual_dict = await trace_testing_manager.wait_for_test_dict()

        with open(json_path, "w") as f:
            json.dump(actual_dict, f)
    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None
