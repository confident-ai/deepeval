from deepeval.integrations.langchain import CallbackHandler
from langchain.chat_models import init_chat_model


def multiply(a: int, b: int) -> int:
    return a * b


llm = init_chat_model("gpt-4.1", model_provider="openai")
llm_with_tools = llm.bind_tools([multiply])

llm_with_tools.invoke(
    "What is 3 * 12?", config={"callbacks": [CallbackHandler()]}
)
