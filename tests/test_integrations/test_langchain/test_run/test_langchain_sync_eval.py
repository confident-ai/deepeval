from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from deepeval.integrations.langchain import CallbackHandler

from deepeval.metrics import TaskCompletionMetric
from tests.test_integrations.utils import assert_trace_json, generate_trace_json
import os


@tool
def multiply(a: int, b: int) -> int:
    """Returns the product of two numbers"""
    return a * b


llm = ChatOpenAI(model="gpt-4o-mini")

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


@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_sync_eval.json"),
    is_run=True,
    ignore_keypaths={
        "root.testCases[": ("toolsCalled",),
    },
)
def test_run_sync_eval():
    from deepeval.dataset import EvaluationDataset, Golden

    dataset = EvaluationDataset(
        goldens=[
            Golden(input="What is 3 * 12?"),
            Golden(input="What is 8 * 6?"),
        ]
    )

    for golden in dataset.evals_iterator():
        agent_executor.invoke(
            {"input": golden.input},
            config={
                "callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]
            },
        )
