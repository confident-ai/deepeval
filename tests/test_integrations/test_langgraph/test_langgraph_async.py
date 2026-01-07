import pytest
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.fake import FakeListLLM
from deepeval.integrations.langchain import CallbackHandler


class State(TypedDict, total=False):
    prompt: str
    output: str


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:The 'config' parameter should be typed as 'RunnableConfig' or 'RunnableConfig \\| None'"
)
async def test_langgraph_async_callback_does_not_print_span_mismatch(capsys):
    llm = FakeListLLM(responses=["pong"])

    async def node(state: State, config=None) -> dict:
        out = await llm.ainvoke(state["prompt"], config=config)
        return {"output": out}

    builder = StateGraph(State)
    builder.add_node("llm", node)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    graph = builder.compile()

    callback = CallbackHandler(metric_collection="test_langgraph_async")

    result = await graph.ainvoke(
        {"prompt": "ping"},
        config={"callbacks": [callback]},
    )

    assert result["output"] == "pong"

    # This is the bug symptom from Observer.__exit__ when contextvars don't match.
    # Pre-fix: you'll see "Current span in context does not match..."
    out = capsys.readouterr().out
    assert (
        "Current span in context does not match the span being exited"
        not in out
    )
