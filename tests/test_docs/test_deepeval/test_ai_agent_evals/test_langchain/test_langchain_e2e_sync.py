import os

from langchain.chat_models import init_chat_model
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric

import deepeval
deepeval.login(os.getenv("CONFIDENT_API_KEY"))

def multiply(a: int, b: int) -> int:
    """Returns the product of two numbers"""
    return a * b

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm_with_tools = llm.bind_tools([multiply])

from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(goldens=[Golden(input="What is 3 * 12?")])
for golden in dataset.evals_iterator():
    llm_with_tools.invoke(
        "What is 3 * 12?",
        config = {"callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]}
    )

