"""
Conditional Routing LangChain App: Routes to different tools based on intent
Complexity: HIGH - Tests conditional logic with deterministic routing

Uses FakeMessagesListChatModel for deterministic tool calls.
"""

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage
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


tools = [research_topic, summarize_text, fact_check]
tools_by_name = {t.name: t for t in tools}


def _create_intent_chain(tool_calls: list, final_message: str):
    """Create a chain for a specific intent."""

    def run_chain(inputs: dict, config: RunnableConfig = None):
        messages = inputs.get("messages", [])

        if tool_calls:
            responses = [
                AIMessage(content="", tool_calls=tool_calls),
                AIMessage(content=final_message),
            ]
        else:
            responses = [AIMessage(content=final_message)]

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

    return RunnableLambda(run_chain)


# Pre-configured chains for each intent
research_chain = _create_intent_chain(
    [
        {
            "name": "research_topic",
            "args": {"topic": "quantum computing"},
            "id": "call_research_001",
            "type": "tool_call",
        }
    ],
    "Based on my research, quantum computing has achieved significant milestones in error correction.",
)

summarize_chain = _create_intent_chain(
    [
        {
            "name": "summarize_text",
            "args": {
                "text": "Artificial intelligence is transforming industries worldwide through automation and data analysis."
            },
            "id": "call_summarize_001",
            "type": "tool_call",
        }
    ],
    "Here is the summary of the text you provided.",
)

fact_check_chain = _create_intent_chain(
    [
        {
            "name": "fact_check",
            "args": {"claim": "The earth is round"},
            "id": "call_fact_001",
            "type": "tool_call",
        }
    ],
    "The fact check confirms that the claim is accurate.",
)

general_chain = _create_intent_chain(
    [],  # No tools for general
    "Hello! I'm here to help. How can I assist you today?",
)


def invoke_research(inputs: dict, config: RunnableConfig = None):
    """Invoke with research intent."""
    return research_chain.invoke(inputs, config=config)


def invoke_summarize(inputs: dict, config: RunnableConfig = None):
    """Invoke with summarize intent."""
    return summarize_chain.invoke(inputs, config=config)


def invoke_fact_check(inputs: dict, config: RunnableConfig = None):
    """Invoke with fact check intent."""
    return fact_check_chain.invoke(inputs, config=config)


def invoke_general(inputs: dict, config: RunnableConfig = None):
    """Invoke with general intent (no tools)."""
    return general_chain.invoke(inputs, config=config)
