"""
Multi-turn Conversation LangGraph Agent with Memory
Complexity: HIGH - Tests conversation history and state persistence
"""

from typing import Literal

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def add_to_cart(item: str, quantity: int = 1) -> str:
    """Add an item to the shopping cart."""
    return f"Added {quantity}x {item} to cart"


@tool
def remove_from_cart(item: str) -> str:
    """Remove an item from the shopping cart."""
    return f"Removed {item} from cart"


@tool
def view_cart() -> str:
    """View the current shopping cart contents."""
    return "Cart: 2x Apple, 1x Banana, 3x Orange"


@tool
def apply_coupon(code: str) -> str:
    """Apply a coupon code to the cart."""
    coupons = {
        "SAVE10": "10% discount applied",
        "SAVE20": "20% discount applied",
        "FREESHIP": "Free shipping applied",
    }
    return coupons.get(code.upper(), f"Invalid coupon: {code}")


@tool
def checkout() -> str:
    """Proceed to checkout."""
    return "Checkout initiated. Total: $25.99. Confirm to place order."


@tool
def confirm_order() -> str:
    """Confirm and place the order."""
    return "Order #12345 placed successfully! Estimated delivery: 3-5 days."


tools = [
    add_to_cart,
    remove_from_cart,
    view_cart,
    apply_coupon,
    checkout,
    confirm_order,
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: dict) -> dict:
    """Shopping assistant agent."""
    messages = state["messages"]
    system_prompt = HumanMessage(
        content="""You are a helpful shopping assistant. Help users:
        - Add/remove items from their cart
        - View their cart
        - Apply coupons
        - Complete checkout
        Remember the conversation context."""
    )
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}


async def async_agent_node(state: dict) -> dict:
    """Async shopping assistant agent."""
    messages = state["messages"]
    system_prompt = HumanMessage(
        content="""You are a helpful shopping assistant. Help users manage their cart and checkout."""
    )
    response = await llm_with_tools.ainvoke([system_prompt] + messages)
    return {"messages": [response]}


def should_continue(state: dict) -> Literal["tools", "__end__"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def build_app_with_memory():
    """Build app with memory checkpointer for multi-turn conversations."""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def build_async_app_with_memory():
    """Build async app with memory checkpointer."""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", async_agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def build_stateless_app():
    """Build stateless app (no memory) for comparison."""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# Export builder functions for tests that need fresh instances
# Pre-built stateless app is safe to reuse
stateless_app = build_stateless_app()


# For memory-based apps, use the builder functions to get fresh instances
# This prevents state leakage between tests
def get_app_with_memory():
    """Get a fresh app instance with memory (use this in tests)."""
    return build_app_with_memory()


def get_async_app_with_memory():
    """Get a fresh async app instance with memory (use this in tests)."""
    return build_async_app_with_memory()
