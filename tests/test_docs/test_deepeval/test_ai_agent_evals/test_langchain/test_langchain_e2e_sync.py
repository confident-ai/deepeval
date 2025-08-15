from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from deepeval.integrations.langchain import CallbackHandler

from deepeval.metrics import TaskCompletionMetric

task_completion_metric = TaskCompletionMetric()


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

from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What is 3 * 12?"),
        Golden(input="What is 8 * 6?"),
        Golden(input="What is 12 * 12?"),
        Golden(input="What is 5 * 5?"),
    ]
)


def llm_agent_eval(golden: Golden):
    result = agent_executor.invoke(
        {"input": golden.input},
        config={
            "callbacks": [CallbackHandler(metrics=[task_completion_metric])]
        },
    )
    return result


for golden in dataset.evals_iterator():
    llm_agent_eval(golden)
