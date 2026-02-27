"""
Metric Collection LangGraph App: Tests metric_collection on LLM and tool spans
Complexity: LOW - Tests metric_collection tracing in LangGraph

Uses ChatOpenAI with metric_collection in metadata and the patched @tool decorator
with metric_collection for component-level evaluations.
"""

from typing import Literal

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from deepeval.integrations.langchain import tool
from deepeval.prompt import Prompt

# Create a Prompt object for prompt tracking
test_prompt = Prompt(alias="langgraph-metric-collection-prompt")
test_prompt.version = "02.00.00"
test_prompt.label = "test-label"
test_prompt.hash = "bab04ec"


@tool(metric_collection="tool_performance")
def convert_temperature(celsius: float) -> str:
    """Converts a temperature from Celsius to Fahrenheit."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C = {fahrenheit}°F"


# LLM with metric_collection and prompt in metadata
llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=42,
    metadata={
        "metric_collection": "llm_accuracy",
        "prompt": test_prompt,
    },
)

llm_with_tools = llm.bind_tools([convert_temperature])
tools = [convert_temperature]


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
    """Build and compile the metric collection agent graph."""
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
