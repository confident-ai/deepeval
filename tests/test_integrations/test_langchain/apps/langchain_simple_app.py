"""
Simple LangChain App: LLM-only, no tools
Complexity: LOW - Tests basic LLM invocation

Uses RunnableLambda wrapper to ensure proper callback events for tracing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig, RunnableLambda

# LLM with deterministic settings
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42)


def _run_simple_chain(messages: list, config: RunnableConfig = None):
    """Run the simple LLM chain."""
    response = llm.invoke(messages, config=config)
    return {"messages": list(messages) + [response]}


async def _arun_simple_chain(messages: list, config: RunnableConfig = None):
    """Async run the simple LLM chain."""
    response = await llm.ainvoke(messages, config=config)
    return {"messages": list(messages) + [response]}


# Wrap as RunnableLambda chains for proper callback event propagation
_simple_chain = RunnableLambda(_run_simple_chain).with_config(
    run_name="simple_chain"
)
_simple_async_chain = RunnableLambda(_arun_simple_chain).with_config(
    run_name="simple_chain"
)


def invoke_simple_app(messages: list, config: RunnableConfig = None):
    """Invoke the simple LLM app with messages."""
    return _simple_chain.invoke(messages, config=config)


async def ainvoke_simple_app(messages: list, config: RunnableConfig = None):
    """Async invoke the simple LLM app with messages."""
    return await _simple_async_chain.ainvoke(messages, config=config)
