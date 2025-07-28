from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric
from langchain.chat_models import init_chat_model
from deepeval.evaluate import dataset
from deepeval.dataset import Golden


def multiply(a: int, b: int) -> int:
    return a * b


llm = init_chat_model("gpt-4.1", model_provider="openai")
llm_with_tools = llm.bind_tools([multiply])

for golden in dataset(goldens=[Golden(input="This is a test query")]):
    llm_with_tools.invoke(
        "What is 3 * 12?",
        config={
            "callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]
        },
    )
