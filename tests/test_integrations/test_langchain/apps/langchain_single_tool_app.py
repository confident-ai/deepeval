"""
Single Tool LangChain App: LLM with one tool
Complexity: LOW - Tests basic tool calling with ChatOpenAI

Uses RunnableLambda wrapper to ensure proper callback events for tracing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig, RunnableLambda


@tool
def get_weather(city: str) -> str:
    """Returns the current weather in a city."""
    weather_data = {
        "san francisco": "Foggy, 58F",
        "new york": "Sunny, 72F",
        "london": "Rainy, 55F",
    }
    return weather_data.get(
        city.lower(), f"Weather data not available for {city}"
    )


tools = [get_weather]
tools_by_name = {t.name: t for t in tools}

# LLM with tool binding
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42)
llm_with_tools = llm.bind_tools(tools)


def _run_tool_chain(inputs: dict, config: RunnableConfig = None):
    """
    Sync tool chain execution:
    1. Call LLM to get tool calls
    2. Execute tools (with proper tool_call structure for callbacks)
    3. Call LLM with tool results
    """
    messages = inputs.get("messages", [])

    # First LLM call
    response = llm_with_tools.invoke(messages, config=config)
    messages_with_response = list(messages) + [response]

    # Execute tool calls if present
    if hasattr(response, "tool_calls") and response.tool_calls:
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
                # Result is a ToolMessage when invoked with tool_call structure
                if isinstance(result, ToolMessage):
                    messages_with_response.append(result)
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)

        # Second LLM call with tool results
        final_response = llm_with_tools.invoke(
            messages_with_response, config=config
        )
        return {"messages": messages_with_response + [final_response]}

    return {"messages": messages_with_response}


async def _arun_tool_chain(inputs: dict, config: RunnableConfig = None):
    """Async tool chain execution."""
    messages = inputs.get("messages", [])

    # First LLM call
    response = await llm_with_tools.ainvoke(messages, config=config)
    messages_with_response = list(messages) + [response]

    # Execute tool calls if present
    if hasattr(response, "tool_calls") and response.tool_calls:
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
                # Result is a ToolMessage when invoked with tool_call structure
                if isinstance(result, ToolMessage):
                    messages_with_response.append(result)
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)

        # Second LLM call with tool results
        final_response = await llm_with_tools.ainvoke(
            messages_with_response, config=config
        )
        return {"messages": messages_with_response + [final_response]}

    return {"messages": messages_with_response}


# Wrap as RunnableLambda chains for proper callback event propagation
_sync_chain = RunnableLambda(_run_tool_chain).with_config(
    run_name="single_tool_chain"
)
_async_chain = RunnableLambda(_arun_tool_chain).with_config(
    run_name="single_tool_chain"
)


def invoke_single_tool_app(inputs: dict, config: RunnableConfig = None):
    """Invoke the single tool app."""
    return _sync_chain.invoke(inputs, config=config)


async def ainvoke_single_tool_app(inputs: dict, config: RunnableConfig = None):
    """Async invoke the single tool app."""
    return await _async_chain.ainvoke(inputs, config=config)
