"""
Streaming LangChain App: LLM with streaming responses
Complexity: MEDIUM - Tests streaming with tool calls

Uses FakeMessagesListChatModel for deterministic tool calls.
"""

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig, RunnableLambda


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
tools_by_name = {t.name: t for t in tools}


def create_streaming_chain(tool_calls_config: list, final_message: str):
    """Create a streaming chain with tool calls."""

    def run_chain(inputs: dict, config: RunnableConfig = None):
        messages = inputs.get("messages", [])

        responses = [
            AIMessage(content="", tool_calls=tool_calls_config),
            AIMessage(content=final_message),
        ]
        llm = FakeMessagesListChatModel(responses=responses)

        # First LLM call
        response = llm.invoke(messages, config=config)
        messages_with_response = list(messages) + [response]

        # Execute tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
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
                    messages_with_response.append(tool_msg)

            # Second LLM call
            final_response = llm.invoke(messages_with_response, config=config)
            return {"messages": messages_with_response + [final_response]}

        return {"messages": messages_with_response}

    def stream_chain(inputs: dict, config: RunnableConfig = None):
        """Stream version that yields chunks."""
        messages = inputs.get("messages", [])

        responses = [
            AIMessage(content="", tool_calls=tool_calls_config),
            AIMessage(content=final_message),
        ]
        llm = FakeMessagesListChatModel(responses=responses)

        # First call - get tool calls
        response = llm.invoke(messages, config=config)
        yield {"agent": response}

        messages_with_response = list(messages) + [response]

        # Execute tools
        if hasattr(response, "tool_calls") and response.tool_calls:
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
                    messages_with_response.append(tool_msg)
                    yield {"tools": tool_msg}

            # Stream final response
            for chunk in llm.stream(messages_with_response, config=config):
                yield {"agent": chunk}

    return RunnableLambda(run_chain), stream_chain


# Single tool streaming
single_stock_calls = [
    {
        "name": "get_stock_price",
        "args": {"symbol": "MSFT"},
        "id": "call_001",
        "type": "tool_call",
    },
]
single_stock_chain, stream_single_stock = create_streaming_chain(
    single_stock_calls, "MSFT is currently trading at $378.90 (+0.8%)."
)

# Multi-tool streaming
multi_stock_calls = [
    {
        "name": "get_stock_price",
        "args": {"symbol": "TSLA"},
        "id": "call_001",
        "type": "tool_call",
    },
    {
        "name": "get_company_info",
        "args": {"symbol": "TSLA"},
        "id": "call_002",
        "type": "tool_call",
    },
]
multi_stock_chain, stream_multi_stock = create_streaming_chain(
    multi_stock_calls, "TSLA (Tesla) is at $245.60. Market cap: $780B."
)


def invoke_streaming_single(inputs: dict, config: RunnableConfig = None):
    """Invoke streaming chain with single tool."""
    return single_stock_chain.invoke(inputs, config=config)


def stream_streaming_single(inputs: dict, config: RunnableConfig = None):
    """Stream with single tool."""
    return stream_single_stock(inputs, config=config)


def invoke_streaming_multi(inputs: dict, config: RunnableConfig = None):
    """Invoke streaming chain with multiple tools."""
    return multi_stock_chain.invoke(inputs, config=config)


def stream_streaming_multi(inputs: dict, config: RunnableConfig = None):
    """Stream with multiple tools."""
    return stream_multi_stock(inputs, config=config)
