import time
from langchain.chat_models import init_chat_model
from deepeval.integrations.langchain import CallbackHandler


def multiply(a: int, b: int) -> int:
    """Returns the product of two numbers"""
    return a * b


llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm_with_tools = llm.bind_tools([multiply])

llm_with_tools.invoke(
    "What is 3 * 12?", config={"callbacks": [CallbackHandler()]}
)

llm_with_tools.invoke(
    "What is 3 * 12?",
    config={
        "callbacks": [
            CallbackHandler(
                metric_collection="metric-collection-name-with-task-completion"
            )
        ]
    },
)
