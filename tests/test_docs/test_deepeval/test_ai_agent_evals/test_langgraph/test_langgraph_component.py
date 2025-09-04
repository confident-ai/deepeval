from typing import TypedDict
from langchain_openai import ChatOpenAI

# from langchain_core.tools import tool
from deepeval.integrations.langchain import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from deepeval.integrations.langchain import CallbackHandler


# ---------------------------
# Define the tool
# ---------------------------
@tool(metric_collection="test_collection_1")
def get_weather(location: str) -> str:
    """Get the current weather in a location."""
    response = ""
    if location.lower() == "london":
        response = "It's rainy and 18°C in London."
    elif location.lower() == "new york":
        response = "It's sunny and 25°C in New York."
    else:
        response = f"Weather info for {location} is not available."

    return response


# ---------------------------
# Define state
# ---------------------------
class State(TypedDict):
    messages: list


# ---------------------------
# Build nodes
# ---------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini", metadata={"metric_collection": "test_collection_1"}
).bind_tools(
    [get_weather]
)  # pass metrics here


def call_llm(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


tools = ToolNode([get_weather])


def call_tools(state: State):
    tool_messages = tools.invoke(state["messages"])
    return {"messages": state["messages"] + tool_messages}


# ---------------------------
# Graph builder
# ---------------------------
workflow = StateGraph(State)

workflow.add_node("llm", call_llm)
workflow.add_node("tools", call_tools)

workflow.set_entry_point("llm")


# routing logic
def route_messages(state: State):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END


workflow.add_conditional_edges(
    "llm", route_messages, {"tools": "tools", END: END}
)
workflow.add_edge("tools", "llm")

app = workflow.compile()


# ---------------------------
# Run the app
# ---------------------------
if __name__ == "__main__":
    inputs = {
        "messages": [
            {"role": "user", "content": "What is the weather in London?"}
        ]
    }
    result = app.invoke(
        inputs,
        config={
            "callbacks": [
                CallbackHandler(metric_collection="test_collection_1")
            ]
        },
    )
    # for m in result["messages"]:
    #     print(m)
