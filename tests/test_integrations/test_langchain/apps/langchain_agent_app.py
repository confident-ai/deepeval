"""
Agent-style LangChain App: Agent that iteratively calls tools
Complexity: HIGH - Tests agent loop with multiple tool calls

Uses ChatOpenAI for live agent behavior with tool calling.
Uses RunnableLambda wrapper to ensure proper callback events for tracing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
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


# Tool sets for different agent configurations
simple_tools = [search_web]
simple_tools_by_name = {t.name: t for t in simple_tools}

multi_step_tools = [search_web, calculator]
multi_step_tools_by_name = {t.name: t for t in multi_step_tools}

complex_tools = [search_web, calculator, get_current_time]
complex_tools_by_name = {t.name: t for t in complex_tools}

# LLM with tool bindings
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42)
llm_simple = llm.bind_tools(simple_tools)
llm_multi_step = llm.bind_tools(multi_step_tools)
llm_complex = llm.bind_tools(complex_tools)


def _run_agent_loop(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
    max_iterations: int = 5,
):
    """Run agent loop synchronously."""
    messages = inputs.get("messages", [])
    all_messages = list(messages)

    for iteration in range(max_iterations):
        # Get next action from LLM
        response = llm_with_tools.invoke(all_messages, config=config)
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
                # Use full tool_call structure to trigger proper callbacks
                tool_call_input = {
                    "name": tool_name,
                    "args": tool_args,
                    "id": tool_id,
                    "type": "tool_call",
                }
                result = tools_by_name[tool_name].invoke(
                    tool_call_input, config=config
                )
                if isinstance(result, ToolMessage):
                    all_messages.append(result)
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    all_messages.append(tool_msg)

    return {"messages": all_messages}


async def _arun_agent_loop(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
    max_iterations: int = 5,
):
    """Run agent loop asynchronously."""
    messages = inputs.get("messages", [])
    all_messages = list(messages)

    for iteration in range(max_iterations):
        # Get next action from LLM
        response = await llm_with_tools.ainvoke(all_messages, config=config)
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
                # Use full tool_call structure to trigger proper callbacks
                tool_call_input = {
                    "name": tool_name,
                    "args": tool_args,
                    "id": tool_id,
                    "type": "tool_call",
                }
                result = await tools_by_name[tool_name].ainvoke(
                    tool_call_input, config=config
                )
                if isinstance(result, ToolMessage):
                    all_messages.append(result)
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    all_messages.append(tool_msg)

    return {"messages": all_messages}


# Create wrapper functions for RunnableLambda
def _simple_agent_sync(inputs: dict, config: RunnableConfig = None):
    return _run_agent_loop(
        inputs, llm_simple, simple_tools_by_name, config=config
    )


async def _simple_agent_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_agent_loop(
        inputs, llm_simple, simple_tools_by_name, config=config
    )


def _multi_step_agent_sync(inputs: dict, config: RunnableConfig = None):
    return _run_agent_loop(
        inputs, llm_multi_step, multi_step_tools_by_name, config=config
    )


async def _multi_step_agent_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_agent_loop(
        inputs, llm_multi_step, multi_step_tools_by_name, config=config
    )


def _complex_agent_sync(inputs: dict, config: RunnableConfig = None):
    return _run_agent_loop(
        inputs, llm_complex, complex_tools_by_name, config=config
    )


async def _complex_agent_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_agent_loop(
        inputs, llm_complex, complex_tools_by_name, config=config
    )


# Wrap as RunnableLambda chains for proper callback event propagation
_simple_agent_chain = RunnableLambda(_simple_agent_sync).with_config(
    run_name="simple_agent"
)
_simple_agent_async_chain = RunnableLambda(_simple_agent_async).with_config(
    run_name="simple_agent"
)
_multi_step_agent_chain = RunnableLambda(_multi_step_agent_sync).with_config(
    run_name="multi_step_agent"
)
_multi_step_agent_async_chain = RunnableLambda(
    _multi_step_agent_async
).with_config(run_name="multi_step_agent")
_complex_agent_chain = RunnableLambda(_complex_agent_sync).with_config(
    run_name="complex_agent"
)
_complex_agent_async_chain = RunnableLambda(_complex_agent_async).with_config(
    run_name="complex_agent"
)


# Simple agent functions (one tool: search_web)
def invoke_simple_agent(inputs: dict, config: RunnableConfig = None):
    """Invoke simple agent (one tool available)."""
    return _simple_agent_chain.invoke(inputs, config=config)


async def ainvoke_simple_agent(inputs: dict, config: RunnableConfig = None):
    """Async invoke simple agent."""
    return await _simple_agent_async_chain.ainvoke(inputs, config=config)


# Multi-step agent functions (two tools: search_web, calculator)
def invoke_multi_step_agent(inputs: dict, config: RunnableConfig = None):
    """Invoke multi-step agent (two tools available)."""
    return _multi_step_agent_chain.invoke(inputs, config=config)


async def ainvoke_multi_step_agent(inputs: dict, config: RunnableConfig = None):
    """Async invoke multi-step agent."""
    return await _multi_step_agent_async_chain.ainvoke(inputs, config=config)


# Complex agent functions (three tools: search_web, calculator, get_current_time)
def invoke_complex_agent(inputs: dict, config: RunnableConfig = None):
    """Invoke complex agent (three tools available)."""
    return _complex_agent_chain.invoke(inputs, config=config)


async def ainvoke_complex_agent(inputs: dict, config: RunnableConfig = None):
    """Async invoke complex agent."""
    return await _complex_agent_async_chain.ainvoke(inputs, config=config)
