from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from deepeval.integrations.langchain import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from deepeval.integrations.langchain import CallbackHandler
from deepeval.prompt import Prompt
import os
import json
import pytest
from tests.test_integrations.utils import assert_json_object_structure, load_trace_data
from tests.test_integrations.manager import trace_testing_manager
import asyncio

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"


@tool(metric_collection="test_collection_1")
def multiply(a: int, b: int) -> int:
    """Returns the product of two numbers"""
    return a * b


llm = ChatOpenAI(
    model="gpt-4o-mini",
    metadata={"metric_collection": "test_collection_1", "prompt": prompt},
)

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can perform mathematical operations.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, [multiply], agent_prompt)

agent_executor = AgentExecutor(agent=agent, tools=[multiply], verbose=True)


def execute_agent():
    result = agent_executor.invoke(
        {"input": "What is 8 multiplied by 6?"},
        config={
            "callbacks": [CallbackHandler(metric_collection="task_completion")]
        },
    )
    return result

################################ TESTING CODE #################################

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, 'langchain.json')

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

        with open(json_path, 'w') as f:
            json.dump(actual_dict, f)
    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None
