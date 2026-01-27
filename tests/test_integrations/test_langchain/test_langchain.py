from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from deepeval.integrations.langchain import tool
from langchain_openai import ChatOpenAI
from deepeval.integrations.langchain import CallbackHandler
from deepeval.prompt import Prompt
import os
from tests.test_integrations.utils import (
    assert_trace_json,
)

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


_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "langchain.json")


# @generate_trace_json(json_path)
@assert_trace_json(json_path)
def test_execute_agent():
    agent_executor.invoke(
        {"input": "What is 8 multiplied by 6?"},
        config={
            "callbacks": [CallbackHandler(metric_collection="task_completion")]
        },
    )


if __name__ == "__main__":
    test_execute_agent()
