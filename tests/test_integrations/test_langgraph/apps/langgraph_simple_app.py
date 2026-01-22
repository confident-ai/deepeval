"""
Simple LangGraph Agent: Single tool with basic state management
Complexity: LOW
"""

from typing import Literal

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


@tool
def get_weather(city: str) -> str:
    """Returns the current weather in a city."""
    weather_data = {
        "san francisco": "Foggy, 58°F",
        "new york": "Sunny, 72°F",
        "london": "Rainy, 55°F",
    }
    return weather_data.get(
        city.lower(), f"Weather data not available for {city}"
    )


# LLM with tool binding
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_weather])

tools = [get_weather]


def agent_node(state: dict, config: RunnableConfig) -> dict:
    """Call the LLM with current messages."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


def should_continue(state: dict) -> Literal["tools", "__end__"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def build_app():
    """Build and compile the simple agent graph."""
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
