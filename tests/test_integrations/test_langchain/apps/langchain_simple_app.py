"""
Simple LangChain App: LLM-only, no tools
Complexity: LOW - Tests basic LLM invocation with deterministic responses
"""

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig


def create_simple_llm():
    """Create a deterministic LLM that returns fixed responses."""
    responses = [
        AIMessage(
            content="Hello! I'm doing well, thank you for asking. How can I help you today?"
        ),
    ]
    return FakeMessagesListChatModel(responses=responses)


llm = create_simple_llm()


def invoke_simple_app(messages: list, config: RunnableConfig = None):
    """Invoke the simple LLM app with messages."""
    return llm.invoke(messages, config=config)


async def ainvoke_simple_app(messages: list, config: RunnableConfig = None):
    """Async invoke the simple LLM app with messages."""
    return await llm.ainvoke(messages, config=config)
