"""
Streaming LangChain App: LLM with streaming responses
Complexity: MEDIUM - Tests streaming with tool calls

Uses RunnableLambda wrapper to ensure proper callback events for tracing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
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


# Single tool setup
single_tools = [get_stock_price]
single_tools_by_name = {t.name: t for t in single_tools}

# Multi tool setup
multi_tools = [get_stock_price, get_company_info]
multi_tools_by_name = {t.name: t for t in multi_tools}

# Streaming LLMs
llm_streaming = ChatOpenAI(
    model="gpt-5-mini", temperature=0, seed=42, streaming=True
)
llm_single = llm_streaming.bind_tools(single_tools)
llm_multi = llm_streaming.bind_tools(multi_tools)


def _run_streaming_chain(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Run a streaming tool chain (invoke mode)."""
    messages = inputs.get("messages", [])

    response = llm_with_tools.invoke(messages, config=config)
    messages_with_response = list(messages) + [response]

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
                if isinstance(result, ToolMessage):
                    messages_with_response.append(result)
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)

        final_response = llm_with_tools.invoke(
            messages_with_response, config=config
        )
        return {"messages": messages_with_response + [final_response]}

    return {"messages": messages_with_response}


async def _arun_streaming_chain(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Async run a streaming tool chain."""
    messages = inputs.get("messages", [])

    response = await llm_with_tools.ainvoke(messages, config=config)
    messages_with_response = list(messages) + [response]

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
                if isinstance(result, ToolMessage):
                    messages_with_response.append(result)
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)

        final_response = await llm_with_tools.ainvoke(
            messages_with_response, config=config
        )
        return {"messages": messages_with_response + [final_response]}

    return {"messages": messages_with_response}


def stream_chain(
    messages: list,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Stream version that yields chunks."""
    response = llm_with_tools.invoke(messages, config=config)
    yield {"agent": response}

    messages_with_response = list(messages) + [response]

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
                if isinstance(result, ToolMessage):
                    messages_with_response.append(result)
                    yield {"tools": result}
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)
                    yield {"tools": tool_msg}

        for chunk in llm_with_tools.stream(
            messages_with_response, config=config
        ):
            yield {"agent": chunk}


async def astream_chain(
    messages: list,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Async stream version that yields chunks."""
    response = await llm_with_tools.ainvoke(messages, config=config)
    yield {"agent": response}

    messages_with_response = list(messages) + [response]

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
                if isinstance(result, ToolMessage):
                    messages_with_response.append(result)
                    yield {"tools": result}
                else:
                    tool_msg = ToolMessage(
                        content=str(result), tool_call_id=tool_id
                    )
                    messages_with_response.append(tool_msg)
                    yield {"tools": tool_msg}

        async for chunk in llm_with_tools.astream(
            messages_with_response, config=config
        ):
            yield {"agent": chunk}


# Create wrapper functions for RunnableLambda
def _streaming_single_sync(inputs: dict, config: RunnableConfig = None):
    return _run_streaming_chain(
        inputs, llm_single, single_tools_by_name, config=config
    )


async def _streaming_single_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_streaming_chain(
        inputs, llm_single, single_tools_by_name, config=config
    )


def _streaming_multi_sync(inputs: dict, config: RunnableConfig = None):
    return _run_streaming_chain(
        inputs, llm_multi, multi_tools_by_name, config=config
    )


async def _streaming_multi_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_streaming_chain(
        inputs, llm_multi, multi_tools_by_name, config=config
    )


# Wrap as RunnableLambda chains for proper callback event propagation
_streaming_single_chain = RunnableLambda(_streaming_single_sync).with_config(
    run_name="streaming_single_chain"
)
_streaming_single_async_chain = RunnableLambda(
    _streaming_single_async
).with_config(run_name="streaming_single_chain")
_streaming_multi_chain = RunnableLambda(_streaming_multi_sync).with_config(
    run_name="streaming_multi_chain"
)
_streaming_multi_async_chain = RunnableLambda(
    _streaming_multi_async
).with_config(run_name="streaming_multi_chain")


# Single tool functions
def invoke_streaming_single(inputs: dict, config: RunnableConfig = None):
    """Invoke streaming chain with single tool."""
    return _streaming_single_chain.invoke(inputs, config=config)


async def ainvoke_streaming_single(inputs: dict, config: RunnableConfig = None):
    """Async invoke streaming chain with single tool."""
    return await _streaming_single_async_chain.ainvoke(inputs, config=config)


def stream_streaming_single(inputs: dict, config: RunnableConfig = None):
    """Stream with single tool."""
    messages = inputs.get("messages", [])
    return stream_chain(
        messages, llm_single, single_tools_by_name, config=config
    )


async def astream_streaming_single(inputs: dict, config: RunnableConfig = None):
    """Async stream with single tool."""
    messages = inputs.get("messages", [])
    async for chunk in astream_chain(
        messages, llm_single, single_tools_by_name, config=config
    ):
        yield chunk


# Multi tool functions
def invoke_streaming_multi(inputs: dict, config: RunnableConfig = None):
    """Invoke streaming chain with multiple tools."""
    return _streaming_multi_chain.invoke(inputs, config=config)


async def ainvoke_streaming_multi(inputs: dict, config: RunnableConfig = None):
    """Async invoke streaming chain with multiple tools."""
    return await _streaming_multi_async_chain.ainvoke(inputs, config=config)


def stream_streaming_multi(inputs: dict, config: RunnableConfig = None):
    """Stream with multiple tools."""
    messages = inputs.get("messages", [])
    return stream_chain(messages, llm_multi, multi_tools_by_name, config=config)


async def astream_streaming_multi(inputs: dict, config: RunnableConfig = None):
    """Async stream with multiple tools."""
    messages = inputs.get("messages", [])
    async for chunk in astream_chain(
        messages, llm_multi, multi_tools_by_name, config=config
    ):
        yield chunk
