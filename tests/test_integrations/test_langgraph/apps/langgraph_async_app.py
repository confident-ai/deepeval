"""
Async LangGraph Agent
Complexity: MEDIUM - Tests async invocation and context propagation
"""

from typing import Literal

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


@tool
def search_database(query: str) -> str:
    """Searches the database for information matching the query."""
    results = {
        "python": "Python is a high-level programming language.",
        "javascript": "JavaScript is a scripting language for web development.",
        "rust": "Rust is a systems programming language focused on safety.",
        "go": "Go is a statically typed language designed at Google.",
    }
    query_lower = query.lower()
    for key, value in results.items():
        if key in query_lower:
            return value
    return f"No results found for: {query}"


@tool
def translate(text: str, target_language: str) -> str:
    """Translates text to the target language (mock)."""
    translations = {
        "spanish": f"[Spanish translation of: {text}]",
        "french": f"[French translation of: {text}]",
        "german": f"[German translation of: {text}]",
    }
    return translations.get(
        target_language.lower(),
        f"Translation to {target_language} not supported",
    )


tools = [search_database, translate]

llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42)
llm_with_tools = llm.bind_tools(tools)


async def agent_node(state: dict, config: RunnableConfig) -> dict:
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


def build_app():
    """Build and compile the async agent graph."""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


app = build_app()
