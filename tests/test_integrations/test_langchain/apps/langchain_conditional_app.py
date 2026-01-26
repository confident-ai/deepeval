"""
Conditional Routing LangChain App: Routes to different tools based on intent
Complexity: HIGH - Tests conditional logic with ChatOpenAI

Uses RunnableLambda wrapper to ensure proper callback events for tracing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig, RunnableLambda


@tool
def research_topic(topic: str) -> str:
    """Research a topic and return findings."""
    research_data = {
        "ai": "AI research shows rapid advancement in large language models and neural networks.",
        "climate": "Climate research indicates rising global temperatures and sea levels.",
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
    if (
        "true" in claim.lower()
        or "correct" in claim.lower()
        or "round" in claim.lower()
    ):
        return "Fact check: VERIFIED - This claim appears to be accurate."
    elif "false" in claim.lower() or "wrong" in claim.lower():
        return "Fact check: FALSE - This claim is inaccurate."
    return "Fact check: UNVERIFIED - Unable to confirm this claim."


# Different tool sets for different intents
research_tools = [research_topic]
research_tools_by_name = {t.name: t for t in research_tools}

summarize_tools = [summarize_text]
summarize_tools_by_name = {t.name: t for t in summarize_tools}

fact_check_tools = [fact_check]
fact_check_tools_by_name = {t.name: t for t in fact_check_tools}

# LLMs
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42)
llm_research = llm.bind_tools(research_tools)
llm_summarize = llm.bind_tools(summarize_tools)
llm_fact_check = llm.bind_tools(fact_check_tools)


def _run_conditional_chain(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Run a conditional tool chain."""
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


async def _arun_conditional_chain(
    inputs: dict,
    llm_with_tools,
    tools_by_name: dict,
    config: RunnableConfig = None,
):
    """Async run a conditional tool chain."""
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


# Create wrapper functions for RunnableLambda
def _research_sync(inputs: dict, config: RunnableConfig = None):
    return _run_conditional_chain(
        inputs, llm_research, research_tools_by_name, config=config
    )


async def _research_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_conditional_chain(
        inputs, llm_research, research_tools_by_name, config=config
    )


def _summarize_sync(inputs: dict, config: RunnableConfig = None):
    return _run_conditional_chain(
        inputs, llm_summarize, summarize_tools_by_name, config=config
    )


async def _summarize_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_conditional_chain(
        inputs, llm_summarize, summarize_tools_by_name, config=config
    )


def _fact_check_sync(inputs: dict, config: RunnableConfig = None):
    return _run_conditional_chain(
        inputs, llm_fact_check, fact_check_tools_by_name, config=config
    )


async def _fact_check_async(inputs: dict, config: RunnableConfig = None):
    return await _arun_conditional_chain(
        inputs, llm_fact_check, fact_check_tools_by_name, config=config
    )


def _general_sync(inputs: dict, config: RunnableConfig = None):
    """General response (no tools)."""
    messages = inputs.get("messages", [])
    response = llm.invoke(messages, config=config)
    return {"messages": list(messages) + [response]}


async def _general_async(inputs: dict, config: RunnableConfig = None):
    """Async general response (no tools)."""
    messages = inputs.get("messages", [])
    response = await llm.ainvoke(messages, config=config)
    return {"messages": list(messages) + [response]}


# Wrap as RunnableLambda chains for proper callback event propagation
_research_chain = RunnableLambda(_research_sync).with_config(
    run_name="research_chain"
)
_research_async_chain = RunnableLambda(_research_async).with_config(
    run_name="research_chain"
)
_summarize_chain = RunnableLambda(_summarize_sync).with_config(
    run_name="summarize_chain"
)
_summarize_async_chain = RunnableLambda(_summarize_async).with_config(
    run_name="summarize_chain"
)
_fact_check_chain = RunnableLambda(_fact_check_sync).with_config(
    run_name="fact_check_chain"
)
_fact_check_async_chain = RunnableLambda(_fact_check_async).with_config(
    run_name="fact_check_chain"
)
_general_chain = RunnableLambda(_general_sync).with_config(
    run_name="general_chain"
)
_general_async_chain = RunnableLambda(_general_async).with_config(
    run_name="general_chain"
)


# Research functions
def invoke_research(inputs: dict, config: RunnableConfig = None):
    """Invoke with research intent."""
    return _research_chain.invoke(inputs, config=config)


async def ainvoke_research(inputs: dict, config: RunnableConfig = None):
    """Async invoke with research intent."""
    return await _research_async_chain.ainvoke(inputs, config=config)


# Summarize functions
def invoke_summarize(inputs: dict, config: RunnableConfig = None):
    """Invoke with summarize intent."""
    return _summarize_chain.invoke(inputs, config=config)


async def ainvoke_summarize(inputs: dict, config: RunnableConfig = None):
    """Async invoke with summarize intent."""
    return await _summarize_async_chain.ainvoke(inputs, config=config)


# Fact check functions
def invoke_fact_check(inputs: dict, config: RunnableConfig = None):
    """Invoke with fact check intent."""
    return _fact_check_chain.invoke(inputs, config=config)


async def ainvoke_fact_check(inputs: dict, config: RunnableConfig = None):
    """Async invoke with fact check intent."""
    return await _fact_check_async_chain.ainvoke(inputs, config=config)


# General functions (no tools)
def invoke_general(inputs: dict, config: RunnableConfig = None):
    """Invoke with general intent (no tools)."""
    return _general_chain.invoke(inputs, config=config)


async def ainvoke_general(inputs: dict, config: RunnableConfig = None):
    """Async invoke with general intent (no tools)."""
    return await _general_async_chain.ainvoke(inputs, config=config)
