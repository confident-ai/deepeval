"""deepeval's LangChain integration on confident-trace, with offline runnables."""

import json
from typing import TypedDict

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("confident_trace")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeListChatModel,
)
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from deepeval.integrations.langchain import CallbackHandler, tool  # noqa: E402
from deepeval.tracing import next_agent_span, trace  # noqa: E402
from deepeval.tracing.otel.setup import ensure_confident_trace  # noqa: E402
from deepeval.tracing.types import (  # noqa: E402
    AgentSpan,
    LlmSpan,
    RetrieverSpan,
    ToolSpan,
)

from tests.test_integrations.utils import walk_spans  # noqa: E402

from .helpers import RecordingMetric  # noqa: E402

PROMPT = ChatPromptTemplate.from_messages([("user", "{question}")])


@pytest.fixture
def langchain(exporter):
    ensure_confident_trace(("langchain", "langgraph"))
    return exporter


class _Retriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content=f"about {query}")]


def spans_of(collected, span_type):
    return [
        span
        for span in walk_spans(collected.root_spans)
        if isinstance(span, span_type)
    ]


def test_handler_settings_reach_a_local_trace(langchain, async_evaluation):
    trace_metric, llm_metric = RecordingMetric(), RecordingMetric()
    model = FakeListChatModel(
        responses=["Paris"],
        metadata={"metrics": [llm_metric], "metric_collection": "answers"},
    )
    handler = CallbackHandler(
        name="capital", thread_id="thread", tags=["geo"], metrics=[trace_metric]
    )

    answer = (PROMPT | model | StrOutputParser()).invoke(
        {"question": "capital?"}, config={"callbacks": [handler]}
    )

    assert answer == "Paris"
    (collected,) = async_evaluation.traces_to_evaluate
    assert (collected.name, collected.thread_id, collected.tags) == (
        "capital",
        "thread",
        ["geo"],
    )
    assert collected.metrics == [trace_metric]
    assert collected.output == "Paris"
    (llm,) = spans_of(collected, LlmSpan)
    assert (llm.metrics, llm.metric_collection) == ([llm_metric], "answers")


def test_the_invoked_run_is_the_agent(langchain, async_evaluation):
    agent_metric = RecordingMetric()
    chain = PROMPT | FakeListChatModel(responses=["Paris"]) | StrOutputParser()

    with next_agent_span(metrics=[agent_metric]):
        chain.invoke(
            {"question": "capital?"}, config={"callbacks": [CallbackHandler()]}
        )

    (collected,) = async_evaluation.traces_to_evaluate
    (root,) = collected.root_spans
    assert isinstance(root, AgentSpan)
    assert (root.metrics, root.output) == ([agent_metric], "Paris")
    # Nested runs keep the type they record.
    assert spans_of(collected, AgentSpan) == [root]
    (llm,) = spans_of(collected, LlmSpan)
    assert llm.metrics is None


def test_retrievers_and_tools(langchain, ended):
    tool_metric = RecordingMetric()

    @tool(metrics=[tool_metric])
    def lookup(city: str) -> str:
        """Look up the weather."""
        return "sunny"

    with trace():
        _Retriever().invoke(
            "paris",
            config={
                "callbacks": [CallbackHandler()],
                "metadata": {"ls_embedding_provider": "openai"},
            },
        )
        lookup.invoke({"city": "Paris"})

    (collected,) = ended
    (retriever,) = spans_of(collected, RetrieverSpan)
    assert retriever.retrieval_context == ["about paris"]
    assert retriever.embedder == "openai"
    (tool_span,) = spans_of(collected, ToolSpan)
    assert (tool_span.name, tool_span.output) == ("lookup", "sunny")
    assert tool_span.metrics == [tool_metric]


def test_handler_settings_reach_an_exported_trace(langchain, ended):
    handler = CallbackHandler(
        name="capital", thread_id="thread", metadata={"team": "geo"}
    )
    (PROMPT | FakeListChatModel(responses=["Paris"])).invoke(
        {"question": "capital?"}, config={"callbacks": [handler]}
    )

    import confident_trace

    confident_trace.flush()
    (root,) = [
        span for span in langchain.get_finished_spans() if span.parent is None
    ]
    assert root.attributes["confident.trace.name"] == "capital"
    assert root.attributes["confident.trace.thread_id"] == "thread"
    assert json.loads(root.attributes["confident.trace.metadata"]) == {
        "team": "geo"
    }
    assert ended == []


def test_a_langgraph_thread_becomes_the_trace_thread(
    langchain, async_evaluation
):
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        question: str
        answer: str

    def answer(state: State):
        reply = FakeListChatModel(responses=["Paris"]).invoke(state["question"])
        return {"answer": reply.content}

    graph = StateGraph(State)
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)

    graph.compile().invoke(
        {"question": "capital?"}, config={"configurable": {"thread_id": "t1"}}
    )

    (collected,) = async_evaluation.traces_to_evaluate
    assert collected.thread_id == "t1"
    assert [span.integration for span in spans_of(collected, LlmSpan)] == [
        "LangGraph"
    ]
