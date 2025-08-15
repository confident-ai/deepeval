import asyncio
from langgraph.prebuilt import create_react_agent
from deepeval.metrics import TaskCompletionMetric

from deepeval.integrations.langchain import CallbackHandler


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


task_completion_metric = TaskCompletionMetric()

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

task_completion = TaskCompletionMetric(
    threshold=0.7, model="gpt-4o-mini", include_reason=True
)

from deepeval.dataset import Golden, EvaluationDataset

goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]

dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    task = asyncio.create_task(
        agent.ainvoke(
            input={"messages": [{"role": "user", "content": golden.input}]},
            config={"callbacks": [CallbackHandler(metrics=[task_completion])]},
        )
    )
    dataset.evaluate(task)
