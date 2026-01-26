"""
Agent-style LangChain App: Agent that iteratively calls tools
Complexity: HIGH - Tests agent loop with multiple tool calls

Uses FakeMessagesListChatModel for deterministic agent behavior.
"""

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig, RunnableLambda


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    results = {
        "weather san francisco": "San Francisco weather: Foggy, 58F, humidity 75%",
        "population tokyo": "Tokyo population: approximately 13.96 million people",
        "stock price apple": "Apple (AAPL) stock: $178.50, up 1.2%",
        "exchange rate usd eur": "USD to EUR: 1 USD = 0.92 EUR",
    }
    for key, value in results.items():
        if all(word in query.lower() for word in key.split()):
            return value
    return f"Search results for '{query}': No specific data found."


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"Calculation: {expression} = {result}"
        return "Invalid expression"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current time (deterministic for testing)."""
    return "Current time: 2024-01-15 10:30:00 UTC"


tools = [search_web, calculator, get_current_time]
tools_by_name = {t.name: t for t in tools}


def create_agent_executor(planned_responses: list, max_iterations: int = 5):
    """
    Create an agent executor that follows planned responses.

    Args:
        planned_responses: List of AIMessage objects to return in sequence
        max_iterations: Maximum number of iterations
    """

    def run_agent(inputs: dict, config: RunnableConfig = None):
        messages = inputs.get("messages", [])
        all_messages = list(messages)

        # Create LLM with planned responses (no bind_tools needed)
        llm = FakeMessagesListChatModel(responses=list(planned_responses))

        for iteration in range(max_iterations):
            # Get next action from LLM
            response = llm.invoke(all_messages, config=config)
            all_messages.append(response)

            # Check if we have tool calls
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                # No more tool calls - agent is done
                break

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                if tool_name in tools_by_name:
                    result = tools_by_name[tool_name].invoke(
                        tool_args, config=config
                    )
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    all_messages.append(tool_msg)

        return {"messages": all_messages}

    return RunnableLambda(run_agent)


# Pre-configured agent scenarios

# Simple agent: search then respond
simple_agent_responses = [
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_web",
                "args": {"query": "weather san francisco"},
                "id": "agent_call_001",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="Based on my search, San Francisco is currently foggy with a temperature of 58F and 75% humidity."
    ),
]

# Multi-step agent: search, calculate, respond
multi_step_agent_responses = [
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_web",
                "args": {"query": "stock price apple"},
                "id": "agent_call_001",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "178.50 * 100"},
                "id": "agent_call_002",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="Apple stock is at $178.50. If you bought 100 shares, they would be worth $17,850."
    ),
]

# Complex agent: multiple tools in sequence
complex_agent_responses = [
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_current_time",
                "args": {},
                "id": "agent_call_001",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_web",
                "args": {"query": "exchange rate usd eur"},
                "id": "agent_call_002",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "1000 * 0.92"},
                "id": "agent_call_003",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="As of 2024-01-15 10:30:00 UTC, the exchange rate is 1 USD = 0.92 EUR. Converting 1000 USD gives you 920 EUR."
    ),
]

simple_agent = create_agent_executor(simple_agent_responses)
multi_step_agent = create_agent_executor(multi_step_agent_responses)
complex_agent = create_agent_executor(complex_agent_responses)


def invoke_simple_agent(inputs: dict, config: RunnableConfig = None):
    """Invoke simple agent (one tool call)."""
    return simple_agent.invoke(inputs, config=config)


def invoke_multi_step_agent(inputs: dict, config: RunnableConfig = None):
    """Invoke multi-step agent (two tool calls in sequence)."""
    return multi_step_agent.invoke(inputs, config=config)


def invoke_complex_agent(inputs: dict, config: RunnableConfig = None):
    """Invoke complex agent (three tool calls in sequence)."""
    return complex_agent.invoke(inputs, config=config)
