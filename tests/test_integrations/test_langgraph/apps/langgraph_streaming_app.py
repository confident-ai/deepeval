"""
Streaming LangGraph Agent
Complexity: MEDIUM - Tests streaming with tool calls
"""

from typing import Literal

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a ticker symbol."""
    prices = {
        "AAPL": "$178.50 (+1.2%)",
        "GOOGL": "$142.30 (-0.5%)",
        "MSFT": "$378.90 (+0.8%)",
        "TSLA": "$245.60 (+2.1%)",
        "AMZN": "$185.20 (-0.3%)",
    }
    return prices.get(symbol.upper(), f"Stock price not available for {symbol}")


@tool
def get_company_info(symbol: str) -> str:
    """Get company information for a ticker symbol."""
    info = {
        "AAPL": "Apple Inc. - Technology company, Market Cap: $2.8T",
        "GOOGL": "Alphabet Inc. - Technology company, Market Cap: $1.8T",
        "MSFT": "Microsoft Corporation - Technology company, Market Cap: $2.9T",
        "TSLA": "Tesla Inc. - Electric vehicles, Market Cap: $780B",
        "AMZN": "Amazon.com Inc. - E-commerce/Cloud, Market Cap: $1.9T",
    }
    return info.get(symbol.upper(), f"Company info not available for {symbol}")


tools = [get_stock_price, get_company_info]

# Enable streaming
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42, streaming=True)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: dict, config: RunnableConfig) -> dict:
    """Agent node - calls the LLM."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


async def async_agent_node(state: dict, config: RunnableConfig) -> dict:
    """Async agent node - calls the LLM."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages, config=config)
    return {"messages": [response]}


def should_continue(state: dict) -> Literal["tools", "__end__"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def build_sync_app():
    """Build sync streaming app."""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def build_async_app():
    """Build async streaming app."""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", async_agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


sync_app = build_sync_app()
async_app = build_async_app()
