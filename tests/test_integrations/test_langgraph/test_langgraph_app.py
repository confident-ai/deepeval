from langgraph.prebuilt import create_react_agent
from deepeval.integrations.langchain import CallbackHandler, tool
import os
import json
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
)
from deepeval.tracing.trace_test_manager import trace_testing_manager
from langchain_openai import ChatOpenAI
from deepeval.prompt import Prompt

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

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "langgraph.json")


# @generate_trace_json(json_path)
@assert_trace_json(json_path)
def test_execute_agent():
    agent.invoke(
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


if __name__ == "__main__":
    test_execute_agent()
