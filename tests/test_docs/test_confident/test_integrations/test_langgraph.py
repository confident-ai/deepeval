from deepeval.metrics import TaskCompletionMetric
from deepeval.dataset import Golden, EvaluationDataset

import os
import time
from langgraph.prebuilt import create_react_agent

import deepeval
from deepeval.integrations.langchain import CallbackHandler


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

# Create a metric
task_completion = TaskCompletionMetric(
    threshold=0.7, model="gpt-4o-mini", include_reason=True
)

# Create goldens
goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]

dataset = EvaluationDataset(goldens=goldens)

# Run evaluation for each golden
for golden in dataset.evals_iterator():
    agent.invoke(
        input={"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler(metrics=[task_completion])]},
    )

# Invoke your agent with the metric collection name
agent.invoke(
    input={
        "messages": [{"role": "user", "content": "what is the weather in sf"}]
    },
    config={
        "callbacks": [
            CallbackHandler(
                metric_collection="<metric-collection-name-with-task-completion>"
            )
        ]
    },
)
