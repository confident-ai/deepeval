import time
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from deepeval.integrations.langchain import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from deepeval.integrations.langchain import CallbackHandler
from deepeval.prompt import Prompt

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")


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
