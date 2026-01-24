"""
Conditional LangGraph Agent
Complexity: HIGH - Multiple conditional edges and routing logic
"""

from typing import Literal, Annotated, Sequence
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig


class ConditionalState(TypedDict):
    """State for the conditional agent with messages and intent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str


@tool
def research_topic(topic: str) -> str:
    """Research a topic and return findings."""
    research_data = {
        "ai": "AI research shows rapid advancement in large language models.",
        "climate": "Climate research indicates rising global temperatures.",
        "space": "Space research reveals new exoplanets in habitable zones.",
        "quantum": "Quantum computing achieves new milestone in error correction.",
    }
    for key, value in research_data.items():
        if key in topic.lower():
            return value
    return f"Research findings for {topic}: General information available."


@tool
def summarize_text(text: str) -> str:
    """Summarize the given text."""
    if len(text) > 100:
        return f"Summary: {text[:100]}..."
    return f"Summary: {text}"


@tool
def fact_check(claim: str) -> str:
    """Fact check a claim."""
    # Simple mock fact checker
    if "true" in claim.lower() or "correct" in claim.lower():
        return "Fact check: VERIFIED - This claim appears to be accurate."
    elif "false" in claim.lower() or "wrong" in claim.lower():
        return "Fact check: FALSE - This claim is inaccurate."
    return "Fact check: UNVERIFIED - Unable to confirm this claim."


tools = [research_topic, summarize_text, fact_check]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=42)
llm_with_tools = llm.bind_tools(tools)


def classify_intent(state: dict) -> dict:
    """Classify the user's intent to route appropriately."""
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content.lower()

    # Simple intent classification
    if "research" in content or "find" in content or "learn" in content:
        intent = "research"
    elif "summarize" in content or "summary" in content:
        intent = "summarize"
    elif "fact" in content or "check" in content or "verify" in content:
        intent = "fact_check"
    else:
        intent = "general"

    return {"messages": messages, "intent": intent}


def research_node(state: dict, config: RunnableConfig) -> dict:
    """Handle research queries."""
    messages = state["messages"]
    system_prompt = HumanMessage(
        content="You are a research assistant. Use the research_topic tool to find information."
    )
    response = llm_with_tools.invoke([system_prompt] + messages, config=config)
    return {"messages": [response]}


def summarize_node(state: dict, config: RunnableConfig) -> dict:
    """Handle summarization queries."""
    messages = state["messages"]
    system_prompt = HumanMessage(
        content="You are a summarization assistant. Use the summarize_text tool."
    )
    response = llm_with_tools.invoke([system_prompt] + messages, config=config)
    return {"messages": [response]}


def fact_check_node(state: dict, config: RunnableConfig) -> dict:
    """Handle fact checking queries."""
    messages = state["messages"]
    system_prompt = HumanMessage(
        content="You are a fact checker. Use the fact_check tool to verify claims."
    )
    response = llm_with_tools.invoke([system_prompt] + messages, config=config)
    return {"messages": [response]}


def general_node(state: dict, config: RunnableConfig) -> dict:
    """Handle general queries."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


def route_by_intent(
    state: dict,
) -> Literal["research", "summarize", "fact_check", "general"]:
    """Route based on classified intent."""
    return state.get("intent", "general")


def should_continue(state: dict) -> Literal["tools", "__end__"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def route_after_tools(state: dict) -> str:
    """Route back to the appropriate node after tool execution."""
    intent = state.get("intent", "general")
    return intent


def build_app():
    """Build the conditional routing graph."""
    graph = StateGraph(ConditionalState)

    # Add nodes
    graph.add_node("classifier", classify_intent)
    graph.add_node("research", research_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("general", general_node)
    graph.add_node("tools", ToolNode(tools))

    # Entry point
    graph.add_edge(START, "classifier")

    # Route from classifier based on intent
    graph.add_conditional_edges(
        "classifier",
        route_by_intent,
        {
            "research": "research",
            "summarize": "summarize",
            "fact_check": "fact_check",
            "general": "general",
        },
    )

    # Each specialized node can go to tools or end
    for node in ["research", "summarize", "fact_check", "general"]:
        graph.add_conditional_edges(
            node, should_continue, {"tools": "tools", "__end__": END}
        )

    # After tools, route back based on intent
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "research": "research",
            "summarize": "summarize",
            "fact_check": "fact_check",
            "general": "general",
        },
    )

    return graph.compile()


app = build_app()
