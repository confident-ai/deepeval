"""
Single Tool LangChain App: LLM with one tool
Complexity: LOW - Tests basic tool calling with deterministic responses

Uses FakeMessagesListChatModel for deterministic tool calls.
"""

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
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


def create_single_tool_chain():
    """
    Create a chain that:
    1. Calls LLM (which emits a tool call)
    2. Executes the tool
    3. Calls LLM again with the tool result
    4. Returns the final response
    """
    # Define responses: first call returns tool call, second returns final answer
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "San Francisco"},
                    "id": "call_weather_001",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="Based on the weather data, San Francisco is currently foggy with a temperature of 58F."
        ),
    ]

    def run_chain(inputs: dict, config: RunnableConfig = None):
        messages = inputs.get("messages", [])

        # Create fresh LLM for each invocation (no bind_tools needed - tool_calls are in responses)
        llm = FakeMessagesListChatModel(responses=list(responses))

        # First LLM call - get tool calls
        response = llm.invoke(messages, config=config)
        messages_with_response = list(messages) + [response]

        # Execute tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                # Find and execute the tool
                if tool_name in tools_by_name:
                    result = tools_by_name[tool_name].invoke(
                        tool_args, config=config
                    )
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)

            # Second LLM call with tool results
            final_response = llm.invoke(messages_with_response, config=config)
            return {"messages": messages_with_response + [final_response]}

        return {"messages": messages_with_response}

    return RunnableLambda(run_chain)


chain = create_single_tool_chain()


def invoke_single_tool_app(inputs: dict, config: RunnableConfig = None):
    """Invoke the single tool app."""
    return chain.invoke(inputs, config=config)


async def ainvoke_single_tool_app(inputs: dict, config: RunnableConfig = None):
    """Async invoke the single tool app."""
    return await chain.ainvoke(inputs, config=config)
