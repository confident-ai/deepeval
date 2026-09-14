"""Public framework calls exercise the unpublished emitter, with no model API."""

import asyncio
import importlib.util
import json

import pytest

pytest.importorskip("confident_trace._core.attachment")
pytest.importorskip("wrapt")

from deepeval.tracing import observe, next_tool_span, next_llm_span
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import LlmSpan, ToolSpan


@pytest.fixture
def embedded(pipeline, monkeypatch, tmp_path):
    from confident_trace._core.runtime import Runtime
    from deepeval.tracing.otel import frameworks

    frameworks.shutdown()
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "crewai"))
    monkeypatch.setenv("CREWAI_TELEMETRY_ENABLED", "false")
    if importlib.util.find_spec("crewai_core"):
        from crewai_core.token_manager import TokenManager

        credentials = tmp_path / "credentials"
        credentials.mkdir()
        monkeypatch.setattr(
            TokenManager,
            "_get_secure_storage_path",
            staticmethod(lambda: credentials),
        )
    rt = Runtime(pipeline[3])
    monkeypatch.setattr(frameworks, "_runtime", rt)
    yield rt
    frameworks.shutdown()
    if importlib.util.find_spec("agents"):
        from agents import set_trace_processors

        set_trace_processors([])


def spans():
    def walk(span):
        yield span
        for child in span.children:
            yield from walk(child)

    return [
        span
        for call in trace_manager.post_trace.call_args_list
        for trace in [call.args[0]]
        for root in trace.root_spans
        for span in walk(root)
    ]


def application(name):
    if name == "langchain":
        pytest.importorskip("langchain_core")
        from deepeval.integrations.langchain import CallbackHandler
        from langchain_core.runnables import RunnableLambda
        from langchain_core.tools import tool

        @tool
        def lookup(query: str) -> str:
            """Look up a query locally."""
            return "answer: " + query

        handler = CallbackHandler(tags=["bridge"])
        runnable = RunnableLambda(lambda query: lookup.invoke({"query": query}))
        return lambda value: runnable.invoke(
            value, config={"callbacks": [handler]}
        ), ToolSpan
    if name == "llamaindex":
        pytest.importorskip("llama_index.core")
        from deepeval.integrations.llama_index import instrument_llama_index
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.tools import FunctionTool

        instrument_llama_index(get_dispatcher())
        tool = FunctionTool.from_defaults(
            fn=lambda query: "answer: " + query, name="lookup"
        )
        return lambda value: tool.call(value), ToolSpan
    if name == "crewai":
        pytest.importorskip("crewai")
        from deepeval.integrations.crewai import instrument_crewai, tool

        instrument_crewai()

        @tool
        def lookup(query: str) -> str:
            """Look up a query locally."""
            return "answer: " + query

        return lambda value: lookup.run(query=value), ToolSpan
    if name == "openai_agents":
        pytest.importorskip("agents")
        pytest.importorskip("openinference.instrumentation.openai_agents")
        from deepeval.openai_agents import DeepEvalTracingProcessor
        from agents import (
            set_trace_processors,
            trace,
            agent_span,
            function_span,
        )

        set_trace_processors([DeepEvalTracingProcessor()])

        def run(value):
            with trace("workflow"):
                with agent_span(name="agent"):
                    with function_span(name="lookup", input=value) as span:
                        span.span_data.output = "answer: " + value
            return "answer: " + value

        return run, ToolSpan
    from openai import _base_client

    httpx = getattr(_base_client, "httpx2", None) or _base_client.httpx

    if name == "openai":
        from deepeval.openai import OpenAI
        from deepeval.openai.patch import patch_openai_classes

        patch_openai_classes()

        def response(request):
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "answer",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                },
            )

        client = OpenAI(
            api_key="local",
            http_client=httpx.Client(transport=httpx.MockTransport(response)),
        )
        return lambda value: client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": value}]
        ), LlmSpan
    if name == "anthropic":
        pytest.importorskip("anthropic")
        from anthropic import _base_client

        httpx = getattr(_base_client, "httpx2", None) or _base_client.httpx
        from deepeval.anthropic import Anthropic
        from deepeval.anthropic.patch import patch_anthropic_classes

        patch_anthropic_classes()

        def response(request):
            return httpx.Response(
                200,
                json={
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "answer"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
            )

        client = Anthropic(
            api_key="local",
            http_client=httpx.Client(transport=httpx.MockTransport(response)),
        )
        return lambda value: client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": value}],
        ), LlmSpan
    raise AssertionError(name)


@pytest.mark.parametrize(
    "name",
    [
        "langchain",
        "llamaindex",
        "crewai",
        "openai_agents",
        "openai",
        "anthropic",
    ],
)
def test_public_framework_call(embedded, name):
    call, kind = application(name)

    @observe()
    def run():
        scope = next_llm_span if kind is LlmSpan else next_tool_span
        with scope(metadata={"local": True}):
            call("question")

    run()
    found = [s for s in spans() if isinstance(s, kind)]
    assert len(found) == 1, [(s.name, type(s).__name__) for s in spans()]
    assert found[0].metadata == {"local": True}
    assert found[0].input
    assert found[0].output
    assert (
        found[0].integration
        == {
            "langchain": "LangChain",
            "llamaindex": "LlamaIndex",
            "crewai": "CrewAI",
            "openai_agents": "OpenAI Agents",
            "openai": "OpenAI",
            "anthropic": "Anthropic",
        }[name]
    )


@pytest.mark.parametrize(
    "name",
    [
        "langchain",
        "llamaindex",
        "crewai",
        "openai_agents",
        "openai",
        "anthropic",
    ],
)
@pytest.mark.parametrize(
    "run_async,schedule", [(False, False), (True, False), (True, True)]
)
def test_framework_iterator(embedded, monkeypatch, name, run_async, schedule):
    from deepeval.dataset import EvaluationDataset, Golden
    from deepeval.evaluate.configs import (
        AsyncConfig,
        DisplayConfig,
        CacheConfig,
    )
    from deepeval.test_run import global_test_run_manager
    from .test_iterator import RecordingMetric
    import importlib

    monkeypatch.setattr(
        global_test_run_manager, "save_test_run", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        global_test_run_manager, "wrap_up_test_run", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        importlib.import_module("deepeval.evaluate.inspect_prompt"),
        "maybe_offer_inspect_tui",
        lambda *a, **kw: None,
    )
    call, kind = application(name)
    RecordingMetric.seen = []
    dataset = EvaluationDataset(
        goldens=[Golden(input="first"), Golden(input="second")]
    )

    def work(value):
        scope = next_llm_span if kind is LlmSpan else next_tool_span
        with scope(metrics=[RecordingMetric()]):
            call(value)

    async def async_work(value):
        await asyncio.sleep(0)
        work(value)

    for golden in dataset.evals_iterator(
        metrics=[RecordingMetric()],
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(use_cache=False, write_cache=False),
    ):
        if schedule:
            dataset.evaluate(asyncio.create_task(async_work(golden.input)))
        else:
            work(golden.input)
    assert len(RecordingMetric.seen) == 4, RecordingMetric.seen
    assert sum("first" in str(value) for value, _ in RecordingMetric.seen) == 2
    assert sum("second" in str(value) for value, _ in RecordingMetric.seen) == 2
    assert not trace_manager.active_spans


def test_langgraph_async_parallel_contexts(embedded):
    pytest.importorskip("langgraph")
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict
    from deepeval.integrations.langchain import CallbackHandler
    from langchain_core.tools import tool
    from deepeval.tracing import update_current_span
    from deepeval.tracing.context import current_span_context

    class State(TypedDict):
        query: str
        left: str
        right: str

    @tool
    async def lookup(query: str) -> str:
        """Look up one query."""
        before = current_span_context.get()
        await asyncio.sleep(0)
        assert current_span_context.get() is before
        assert isinstance(before, ToolSpan)
        update_current_span(metadata={"query": query})
        return query

    async def left(state):
        return {
            "left": await lookup.ainvoke({"query": state["query"] + "-left"})
        }

    async def right(state):
        return {
            "right": await lookup.ainvoke({"query": state["query"] + "-right"})
        }

    graph = StateGraph(State)
    graph.add_node("left", left)
    graph.add_node("right", right)
    for node in ("left", "right"):
        graph.add_edge(START, node)
        graph.add_edge(node, END)
    app = graph.compile()
    handler = CallbackHandler(thread_id="conversation", tags=["parallel"])

    @observe()
    async def run(query):
        return await app.ainvoke(
            {"query": query}, config={"callbacks": [handler]}
        )

    async def both():
        await asyncio.gather(run("one"), run("two"))

    asyncio.run(both())
    found = [s for s in spans() if isinstance(s, ToolSpan)]
    assert len(found) == 4
    assert {s.metadata["query"] for s in found} == {
        "one-left",
        "one-right",
        "two-left",
        "two-right",
    }
    assert len({s.trace_uuid for s in found}) == 2
    assert not trace_manager.active_spans


def test_llamaindex_local_model_and_retriever(embedded):
    pytest.importorskip("llama_index.core")
    from llama_index.core.instrumentation import get_dispatcher
    from llama_index.core.llms import ChatMessage, MockLLM
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode
    from deepeval.integrations.llama_index import instrument_llama_index
    from deepeval.tracing.types import RetrieverSpan

    class Retriever(BaseRetriever):
        def _retrieve(self, query_bundle):
            return [NodeWithScore(node=TextNode(text="evidence"), score=1)]

    instrument_llama_index(get_dispatcher())

    @observe()
    def run():
        MockLLM().chat([ChatMessage(role="user", content="question")])
        Retriever().retrieve("question")

    run()
    llms = [s for s in spans() if isinstance(s, LlmSpan)]
    assert len(llms) == 1, [(type(s).__name__, s.name) for s in spans()]
    assert "question" in str(llms[0].input)
    assert llms[0].output
    retrieved = [s for s in spans() if isinstance(s, RetrieverSpan)]
    assert len(retrieved) == 1
    assert "evidence" in str(retrieved[0].output)


def test_crewai_wrapper_metrics_and_reset(embedded):
    pytest.importorskip("crewai")
    from deepeval.integrations.crewai import (
        instrument_crewai,
        reset_crewai_instrumentation,
        tool,
    )
    from .test_iterator import RecordingMetric

    marker = RecordingMetric()
    instrument_crewai()

    @tool(metric=[marker], metric_collection="tools")
    def lookup(query: str) -> str:
        """Look up a query."""
        return query

    @observe()
    def run():
        lookup.run(query="question")

    run()
    found = [s for s in spans() if isinstance(s, ToolSpan)]
    assert len(found) == 1
    assert found[0].metrics == [marker]
    assert found[0].metric_collection == "tools"
    reset_crewai_instrumentation()
    trace_manager.post_trace.reset_mock()
    run()
    assert len([s for s in spans() if isinstance(s, ToolSpan)]) == 1
    instrument_crewai()
    trace_manager.post_trace.reset_mock()
    run()
    assert len([s for s in spans() if isinstance(s, ToolSpan)]) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stop_early", [False, True])
def test_openai_stream_lifecycle(embedded, asynchronous, stop_early):
    from openai import _base_client

    httpx = getattr(_base_client, "httpx2", None) or _base_client.httpx
    from deepeval.openai import OpenAI, AsyncOpenAI
    from deepeval.openai.patch import patch_openai_classes

    patch_openai_classes()

    def response(request):
        chunks = [
            {
                "id": "local",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            }
            for text in ("hel", "lo")
        ]
        body = (
            "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    @observe()
    def sync_run():
        with OpenAI(
            api_key="local",
            http_client=httpx.Client(transport=httpx.MockTransport(response)),
        ) as client:
            with client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "question"}],
                stream=True,
            ) as stream:
                for chunk in stream:
                    if stop_early:
                        break

    @observe()
    async def async_run():
        async with AsyncOpenAI(
            api_key="local",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(response)
            ),
        ) as client:
            async with await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "question"}],
                stream=True,
            ) as stream:
                async for chunk in stream:
                    await asyncio.sleep(0)
                    if stop_early:
                        break

    if asynchronous:
        asyncio.run(async_run())
    else:
        sync_run()
    found = [s for s in spans() if isinstance(s, LlmSpan)]
    assert len(found) == 1
    assert ("hel" if stop_early else "hello") in str(found[0].output)
    assert not trace_manager.active_spans


def test_crewai_native_llm_wrapper_metrics(embedded, monkeypatch):
    pytest.importorskip("crewai")
    from openai import _base_client

    httpx = getattr(_base_client, "httpx2", None) or _base_client.httpx
    import openai
    from deepeval.integrations.crewai import LLM, instrument_crewai
    from .test_iterator import RecordingMetric

    monkeypatch.setenv("OPENAI_API_KEY", "local")
    # Construct before enabling instrumentation, as applications commonly do.
    marker = RecordingMetric()
    model = LLM(
        model="openai/gpt-4o-mini", metrics=[marker], metric_collection="models"
    )
    model.client = openai.OpenAI(
        api_key="local",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "id": "local",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "answer",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "total_tokens": 3,
                        },
                    },
                )
            )
        ),
    )
    monkeypatch.setattr(
        type(model), "_get_sync_client", lambda self: model.client
    )
    instrument_crewai()
    from deepeval.openai.patch import patch_openai_classes

    patch_openai_classes()

    @observe()
    def run():
        return model.call([{"role": "user", "content": "question"}])

    assert run() == "answer"
    found = [s for s in spans() if isinstance(s, LlmSpan)]
    assert len(found) == 1, [(s.name, type(s).__name__) for s in spans()]
    assert found[0].metrics == [marker]
    assert found[0].metric_collection == "models"
    assert found[0].input and found[0].output


def test_openai_agents_runner_wrapper_metrics(embedded):
    pytest.importorskip("agents")
    pytest.importorskip("openinference.instrumentation.openai_agents")
    from agents import Runner, set_trace_processors
    from agents.models.interface import Model
    from agents.items import ModelResponse
    from agents.usage import Usage
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText
    from deepeval.openai_agents import Agent, DeepEvalTracingProcessor
    from .test_iterator import RecordingMetric
    from deepeval.tracing.types import AgentSpan

    class LocalModel(Model):
        async def get_response(self, *args, **kwargs):
            return ModelResponse(
                output=[
                    ResponseOutputMessage(
                        id="local",
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[
                            ResponseOutputText(
                                type="output_text",
                                text="answer",
                                annotations=[],
                            )
                        ],
                    )
                ],
                usage=Usage(
                    requests=1, input_tokens=2, output_tokens=1, total_tokens=3
                ),
                response_id="local",
            )

        async def stream_response(self, *args, **kwargs):
            raise NotImplementedError
            yield

    marker = RecordingMetric()
    set_trace_processors([DeepEvalTracingProcessor()])
    agent = Agent(
        name="agent",
        model=LocalModel(),
        agent_metrics=[marker],
        llm_metrics=[marker],
        llm_metric_collection="models",
    )

    @observe()
    async def run():
        return await Runner.run(agent, "question")

    assert asyncio.run(run()).final_output == "answer"
    llms = [s for s in spans() if isinstance(s, LlmSpan)]
    assert len(llms) == 1
    assert llms[0].metrics == [marker]
    assert llms[0].metric_collection == "models"
    assert any(
        s.metrics == [marker] for s in spans() if isinstance(s, AgentSpan)
    )


def test_langchain_trace_and_component_options(embedded):
    pytest.importorskip("langchain_core")
    from deepeval.integrations.langchain import CallbackHandler
    from deepeval.tracing import next_agent_span
    from deepeval.tracing.types import AgentSpan
    from langchain_core.runnables import RunnableLambda
    from langchain_core.language_models.fake_chat_models import (
        FakeListChatModel,
    )
    from .test_iterator import RecordingMetric

    trace_metric, model_metric, staged_metric = (
        RecordingMetric(),
        RecordingMetric(),
        RecordingMetric(),
    )
    handler = CallbackHandler(
        metrics=[trace_metric],
        metric_collection="trace-metrics",
        name="configured",
    )
    model = FakeListChatModel(responses=["answer"])
    app = RunnableLambda(
        lambda value: model.invoke(
            value,
            config={
                "metadata": {
                    "metrics": [model_metric],
                    "metric_collection": "model-metrics",
                }
            },
        )
    )

    @observe()
    def run():
        with next_agent_span(metadata={"root": True}):
            with next_llm_span(metrics=[staged_metric]):
                app.invoke("question", config={"callbacks": [handler]})

    run()
    trace = trace_manager.post_trace.call_args.args[0]
    assert trace.metrics == [trace_metric]
    assert trace.metric_collection == "trace-metrics"
    assert trace.name == "configured"
    agent = next(s for s in spans() if isinstance(s, AgentSpan))
    assert agent.metadata == {"root": True}
    assert agent.metrics is None
    model_span = next(s for s in spans() if isinstance(s, LlmSpan))
    assert model_span.metrics == [staged_metric]
    assert model_span.metric_collection == "model-metrics"


def test_explicit_langchain_handler_does_not_enroll_other_runs(embedded):
    pytest.importorskip("langchain_core")
    from deepeval.integrations.langchain import CallbackHandler
    from langchain_core.runnables import RunnableLambda

    handler = CallbackHandler()
    app = RunnableLambda(lambda value: value)

    @observe()
    def run():
        app.invoke("without callback")
        app.invoke("with callback", config={"callbacks": [handler]})

    run()
    assert len(spans()) == 2
    assert "with callback" in str(spans()[1].input)


def test_private_embedding_leaves_public_runtime_and_exporter_owned(embedded):
    from confident_trace._core import runtime
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    public_runtime = runtime.current()
    public_provider = otel_trace.get_tracer_provider()
    exporter = InMemorySpanExporter()
    embedded.provider.add_span_processor(SimpleSpanProcessor(exporter))
    call, _ = application("openai")

    @observe()
    def run():
        call("question")

    run()
    assert runtime.current() is public_runtime
    assert otel_trace.get_tracer_provider() is public_provider
    assert len(exporter.get_finished_spans()) == 1
    assert len([s for s in spans() if isinstance(s, LlmSpan)]) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_anthropic_stream_manager(embedded, asynchronous):
    pytest.importorskip("anthropic")
    from anthropic import _base_client
    from deepeval.anthropic import Anthropic, AsyncAnthropic
    from deepeval.anthropic.patch import patch_anthropic_classes

    httpx = getattr(_base_client, "httpx2", None) or _base_client.httpx
    patch_anthropic_classes()
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "local",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 2, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "answer"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]

    def response(request):
        body = "".join(
            "event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n"
            for event in events
        )
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    kwargs = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "question"}],
        "max_tokens": 10,
    }

    @observe()
    def run():
        with Anthropic(
            api_key="local",
            http_client=httpx.Client(transport=httpx.MockTransport(response)),
        ) as client:
            with client.messages.stream(**kwargs) as stream:
                assert "".join(stream.text_stream) == "answer"

    @observe()
    async def arun():
        async with AsyncAnthropic(
            api_key="local",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(response)
            ),
        ) as client:
            async with client.messages.stream(**kwargs) as stream:
                assert (
                    "".join([part async for part in stream.text_stream])
                    == "answer"
                )

    asyncio.run(arun()) if asynchronous else run()
    found = [s for s in spans() if isinstance(s, LlmSpan)]
    assert len(found) == 1
    assert "answer" in str(found[0].output)
    assert found[0].input_token_count == 2
    assert found[0].output_token_count == 1
    assert not trace_manager.active_spans


def test_llamaindex_multiple_dispatchers(embedded):
    pytest.importorskip("llama_index.core")
    from llama_index_instrumentation.dispatcher import Dispatcher
    from deepeval.integrations.llama_index import instrument_llama_index

    first = Dispatcher(name="first", propagate=False)
    second = Dispatcher(name="second", propagate=False)
    instrument_llama_index(first)
    instrument_llama_index(second)
    instrument_llama_index(second)

    @first.span
    def first_call(query):
        return query

    @second.span
    def second_call(query):
        return query

    @observe()
    def run():
        first_call("one")
        second_call("two")

    run()
    assert len(spans()) == 3
    assert len(second.span_handlers) == 1


def test_agents_without_optional_bridge_extra_uses_legacy(
    embedded, monkeypatch
):
    pytest.importorskip("agents")
    import sys
    from deepeval.openai_agents import DeepEvalTracingProcessor

    monkeypatch.setitem(
        sys.modules, "openinference.instrumentation.openai_agents", None
    )
    processor = DeepEvalTracingProcessor()
    assert processor._otel_delegate is None
