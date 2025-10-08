from langgraph.prebuilt import create_react_agent
from deepeval.integrations.langchain import CallbackHandler, tool
import os
import json
import pytest
from tests.test_integrations.utils import (
    assert_json_object_structure,
    load_trace_data,
)
from tests.test_integrations.manager import trace_testing_manager
from langchain_openai import ChatOpenAI
from deepeval.prompt import Prompt
import asyncio

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"


@tool(metric_collection="test_collection_1")
def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


llm = ChatOpenAI(
    model="gpt-4o-mini",
    metadata={"metric_collection": "test_collection_1", "prompt": prompt},
)

agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant",
)


def execute_agent():
    return agent.invoke(
        input={
            "messages": [
                {"role": "user", "content": "what is the weather in sf"}
            ]
        },
        config={
            "callbacks": [
                CallbackHandler(
                    name="langgraph-test",
                    tags=["langgraph", "test"],
                    metadata={"environment": "test"},
                    thread_id="123",
                    user_id="456",
                    metric_collection="task_completion",
                )
            ],
        },
    )


################################ TESTING CODE #################################

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "langgraph.json")


@pytest.mark.asyncio
async def test_json_schema():
    """
    Test the json schema of the trace. Raises an exception if the schema is invalid.
    """
    try:
        trace_testing_manager.test_name = json_path
        execute_agent()
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
        execute_agent()
        actual_dict = await trace_testing_manager.wait_for_test_dict()

        with open(json_path, "w") as f:
            json.dump(actual_dict, f)
    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None
