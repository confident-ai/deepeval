"""LangGraph Next-Span App: validates ``with next_llm_span(...)`` against
a real ``ChatOpenAI`` driving a ``StateGraph`` agent loop.

Mirrors ``test_langchain/apps/langchain_next_span_app.py`` for the
LangGraph orchestration surface. Same handler (``CallbackHandler``) and
same plumbing — what's distinct is that the agent loop is now an
explicit ``StateGraph`` (agent node → tools node → agent node) so the
"first LLM span only" one-shot semantic is visible as a structural
property of the graph (the second agent-node visit MUST emit a chat
model span without ``metric_collection``).

We do NOT bake ``metric_collection`` into ``ChatOpenAI(metadata=...)``
so the staged value has no metadata-level peer to confuse precedence.
"""

from typing import Dict, Literal, Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from deepeval.tracing import next_llm_span


@tool
def square(n: int) -> int:
    """Returns the square of the input integer."""
    return n * n


_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0, seed=42)
_llm_with_tools = _llm.bind_tools([square])
_tools = [square]


def _agent_node(state: dict, config: RunnableConfig) -> dict:
    response = _llm_with_tools.invoke(state["messages"], config=config)
    return {"messages": [response]}


def _should_continue(state: dict) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


def _build_app():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(_tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", _should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


_app = _build_app()


def invoke_with_next_llm_span(
    inputs: dict,
    metric_collection: str,
    metadata: Optional[Dict] = None,
    config: RunnableConfig = None,
):
    """Wrap the graph invocation in ``with next_llm_span(...)``.

    The graph's first agent-node visit triggers the FIRST chat-model
    span — that's the one the staged value lands on. The post-tool
    agent-node visit fires a second chat-model span; the pending slot
    has been drained, so the trace must show ``metric_collection: null``
    on it.
    """
    with next_llm_span(
        metric_collection=metric_collection,
        metadata=metadata,
    ):
        return _app.invoke(inputs, config=config)


async def ainvoke_with_next_llm_span(
    inputs: dict,
    metric_collection: str,
    metadata: Optional[Dict] = None,
    config: RunnableConfig = None,
):
    """Async counterpart. The pending-slot ContextVar must propagate
    through LangGraph's asyncio task scheduling to the chat-model
    callback inside the agent node."""
    with next_llm_span(
        metric_collection=metric_collection,
        metadata=metadata,
    ):
        return await _app.ainvoke(inputs, config=config)
