from langgraph.prebuilt import create_react_agent

from deepeval.integrations.langchain import CallbackHandler

from deepeval.metrics import TaskCompletionMetric

import os
from tests.test_integrations.utils import assert_trace_json, generate_trace_json


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

_current_dir = os.path.dirname(os.path.abspath(__file__))


@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_sync_eval.json"),
    is_run=True,
    ignore_keypaths={
        "root.testCases[": ("toolsCalled",),
    },
)
def test_run_sync_eval():
    from deepeval.dataset import Golden, EvaluationDataset

    goldens = [
        Golden(input="What is the weather in Bogotá, Colombia?"),
        Golden(input="What is the weather in Paris, France?"),
    ]

    dataset = EvaluationDataset(goldens=goldens)

    for golden in dataset.evals_iterator():
        agent.invoke(
            input={"messages": [{"role": "user", "content": golden.input}]},
            config={
                "callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]
            },
        )
