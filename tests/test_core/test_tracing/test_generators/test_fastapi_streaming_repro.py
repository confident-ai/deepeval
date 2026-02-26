"""
Tests for @observe'd sync generators consumed across thread-pool threads.

When an ASGI framework (e.g. FastAPI / Starlette) streams a sync generator,
each next() call is dispatched to a thread-pool thread.  ContextVar.set()
inside one thread does NOT propagate to subsequent threads, so
Observer.__exit__ must fall back to UUID-based lookups and the generator
wrapper must restore ContextVars on each resume.

Run with:
    pytest tests/test_core/test_tracing/test_generators/test_fastapi_streaming_repro.py -xvs

Generate schemas:
    GENERATE_SCHEMAS=true pytest tests/test_core/test_tracing/test_generators/test_fastapi_streaming_repro.py -xvs
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from deepeval.tracing import (
    observe,
    trace,
    update_llm_span,
    update_retriever_span,
)
from deepeval.tracing.tracing import trace_manager
from tests.test_core.test_tracing.conftest import trace_test


# ── leaf spans (retriever + LLM) ──────────────────────────────────────


@observe(type="retriever")
def retrieve_documents(query: str) -> str:
    update_retriever_span(
        embedder="text-embedding-3-small", top_k=5, chunk_size=512
    )
    return f"relevant context for: {query}"


@observe(type="llm", model="gpt-4o")
def stream_llm_tokens(query: str, context: str):
    tokens = [
        f"Based on {context}, ",
        f"the answer to '{query}' ",
        "is 42.",
    ]
    for token in tokens:
        yield token
    update_llm_span(
        input_token_count=len(query.split()) + len(context.split()),
        output_token_count=len(tokens),
        cost_per_input_token=0.005,
        cost_per_output_token=0.015,
    )


@observe(type="llm", model="gpt-4o-mini")
def summarize(text: str) -> str:
    update_llm_span(input_token_count=10, output_token_count=5)
    return f"summary of: {text}"


@observe()
def transform_text(text: str) -> str:
    return f"processed {text}"


# ── composite generators ──────────────────────────────────────────────


@observe()
def streamed_tokens(prompt: str):
    """Simple generator — yields plain strings."""
    yield f"Hello, "
    yield f"you said: "
    yield prompt


@observe()
def streamed_with_child(prompt: str):
    """Generator that calls a child @observe'd function."""
    result = transform_text(prompt)
    for word in result.split():
        yield word


@observe()
def streaming_rag_pipeline(query: str):
    """RAG pipeline: retrieve docs → stream LLM tokens.

    Produces a retriever span, an LLM span (itself a generator), and
    a base span for the pipeline itself — all nested.
    """
    context = retrieve_documents(query)
    yield f"[context] {context}\n"
    for token in stream_llm_tokens(query, context):
        yield token
    yield "[done]\n"


@observe()
def multi_step_pipeline(query: str):
    """Deeper nesting: pipeline → RAG sub-pipeline → leaf spans, plus
    a sibling LLM call for summarization.

    Span tree:
        multi_step_pipeline (base)
        ├─ streaming_rag_pipeline (base)
        │  ├─ retrieve_documents (retriever)
        │  └─ stream_llm_tokens (llm, generator)
        └─ summarize (llm)
    """
    yield {"step": "start"}
    rag_output = []
    for chunk in streaming_rag_pipeline(query):
        rag_output.append(chunk)
        yield {"step": "rag", "chunk": chunk}
    result = summarize(" ".join(rag_output))
    yield {"step": "summary", "result": result}
    yield {"step": "done"}


def streaming_rag_pipeline_plain(query: str):
    """Undecorated version — inner calls still have @observe."""
    context = retrieve_documents(query)
    yield f"[context] {context}\n"
    for token in stream_llm_tokens(query, context):
        yield token
    yield "[done]\n"


# ── endpoint variants (multi-trace scenarios) ─────────────────────────


@observe()
def observed_endpoint(query: str):
    return streaming_rag_pipeline(query)


def trace_wrapped_endpoint(query: str):
    with trace(name="endpoint"):
        return streaming_rag_pipeline(query)


def trace_wrapped_plain_endpoint(query: str):
    with trace(name="endpoint"):
        return streaming_rag_pipeline_plain(query)


# ── thread-pool simulation (mirrors Starlette's iterate_in_threadpool) ──

EXPECTED_RAG_CHUNKS = [
    "[context] relevant context for: hello\n",
    "Based on relevant context for: hello, ",
    "the answer to 'hello' ",
    "is 42.",
    "[done]\n",
]

_STOP = object()


def _next_or_sentinel(gen):
    try:
        return next(gen)
    except StopIteration:
        return _STOP


async def _iterate_in_threadpool(gen):
    """Simulates Starlette's iterate_in_threadpool for sync generators."""
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2)
    chunks = []
    while True:
        ctx = copy_context()
        chunk = await loop.run_in_executor(
            executor, ctx.run, _next_or_sentinel, gen
        )
        if chunk is _STOP:
            break
        chunks.append(chunk)
    executor.shutdown(wait=False)
    return chunks


def run_in_threadpool(gen):
    """Sync wrapper: runs the threadpool simulation and returns chunks."""
    return asyncio.run(_iterate_in_threadpool(gen))


async def _call_then_iterate(endpoint_fn, prompt):
    """Call endpoint in a thread, then iterate the returned generator."""
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2)
    ctx = copy_context()
    gen = await loop.run_in_executor(executor, ctx.run, endpoint_fn, prompt)
    chunks = await _iterate_in_threadpool(gen)
    executor.shutdown(wait=False)
    return chunks


# ── helpers ────────────────────────────────────────────────────────────


def _assert_all_traces_valid():
    """Every trace and span must have end_time set and serialize OK."""
    for i, t in enumerate(trace_manager.traces):
        assert t.end_time is not None, f"trace[{i}] end_time is None"
        for root_span in t.root_spans:
            _assert_all_spans_closed(root_span, trace_idx=i)
        trace_api = trace_manager.create_trace_api(t)
        body = trace_api.model_dump(by_alias=True, exclude_none=True)
        assert isinstance(body["endTime"], str)


def _assert_all_spans_closed(span, trace_idx=0, depth=0):
    """Recursively verify every span in the tree has end_time set."""
    assert span.end_time is not None, (
        f"trace[{trace_idx}] span '{span.name}' (depth={depth}) "
        f"has end_time=None"
    )
    for child in span.children:
        _assert_all_spans_closed(child, trace_idx, depth + 1)


# ── tests: schema-validated (single-trace scenarios) ──────────────────


class TestFastAPIStreamingRepro:

    @trace_test("generators/fastapi_basic_threadpool_schema.json")
    def test_simple_generator_across_threadpool(self):
        """Single @observe'd generator iterated across thread-pool threads."""
        gen = streamed_tokens("world")
        chunks = run_in_threadpool(gen)
        assert chunks == ["Hello, ", "you said: ", "world"]

    @trace_test("generators/fastapi_child_spans_threadpool_schema.json")
    def test_generator_with_child_span_across_threadpool(self):
        """Generator that calls a child @observe'd function across threads."""
        gen = streamed_with_child("hello")
        chunks = run_in_threadpool(gen)
        assert chunks == ["processed", "hello"]

    @trace_test("generators/fastapi_rag_pipeline_threadpool_schema.json")
    def test_rag_pipeline_across_threadpool(self):
        """
        RAG pipeline generator with retriever + LLM child spans,
        iterated across thread-pool threads.  Exercises nested
        generators, mixed span types, and update_llm_span /
        update_retriever_span across thread boundaries.
        """
        gen = streaming_rag_pipeline("hello")
        chunks = run_in_threadpool(gen)
        assert chunks == EXPECTED_RAG_CHUNKS

    @trace_test("generators/fastapi_deep_nesting_threadpool_schema.json")
    def test_deep_nesting_across_threadpool(self):
        """
        4-level nesting across thread-pool threads:
        multi_step_pipeline → streaming_rag_pipeline →
        retrieve_documents + stream_llm_tokens, plus a sibling
        summarize call.
        """
        gen = multi_step_pipeline("hello")
        chunks = run_in_threadpool(gen)
        assert chunks[0] == {"step": "start"}
        assert chunks[-1] == {"step": "done"}

    @trace_test("generators/fastapi_same_thread_sanity_schema.json")
    def test_same_thread_rag_pipeline(self):
        """Sanity check: same-thread consumption of the RAG pipeline."""
        chunks = list(streaming_rag_pipeline("test"))
        assert len(chunks) == 5


# ── tests: multi-trace edge cases (manual assertions) ─────────────────


class TestFastAPIStreamingMultiTrace:
    """
    Scenarios that produce multiple traces (endpoint trace + generator
    trace). These can't use @trace_test which captures a single trace,
    so they use manual assertions instead.
    """

    def test_observe_on_both_endpoint_and_generator(self):
        """
        @observe on both the endpoint and the inner generator.
        The endpoint span finishes immediately (returns generator object),
        the generator creates a second trace when consumed.
        """
        chunks = asyncio.run(_call_then_iterate(observed_endpoint, "hello"))
        assert chunks == EXPECTED_RAG_CHUNKS
        assert len(trace_manager.traces) >= 1
        _assert_all_traces_valid()

    def test_trace_context_wrapping_observed_generator(self):
        """
        with trace() wraps the call site + @observe on the generator.
        The trace context ends immediately; the generator creates a
        second trace when consumed.
        """
        chunks = asyncio.run(
            _call_then_iterate(trace_wrapped_endpoint, "hello")
        )
        assert chunks == EXPECTED_RAG_CHUNKS
        assert len(trace_manager.traces) >= 1
        _assert_all_traces_valid()

    def test_trace_context_with_plain_generator(self):
        """
        with trace() at the call site, but the generator has no @observe.
        The trace context ends immediately; child spans inside the
        generator still create their own traces.
        """
        chunks = asyncio.run(
            _call_then_iterate(trace_wrapped_plain_endpoint, "hello")
        )
        assert chunks == EXPECTED_RAG_CHUNKS
        assert len(trace_manager.traces) >= 1
        _assert_all_traces_valid()

    def test_single_trace_with_nested_hierarchy(self):
        """
        A single @observe'd RAG pipeline generator must produce exactly
        1 trace with correct parent-child hierarchy preserved across
        thread-pool threads.
        """
        gen = streaming_rag_pipeline("hello")
        chunks = run_in_threadpool(gen)

        assert chunks == EXPECTED_RAG_CHUNKS
        assert len(trace_manager.traces) == 1, (
            f"Expected 1 trace but got {len(trace_manager.traces)} — "
            "child spans should be nested, not in separate traces"
        )

        t = trace_manager.traces[0]
        root = t.root_spans[0]
        assert root.name == "streaming_rag_pipeline"
        assert len(root.children) == 2, (
            "retrieve_documents and stream_llm_tokens should be nested "
            "under streaming_rag_pipeline"
        )
        child_names = {c.name for c in root.children}
        assert child_names == {"retrieve_documents", "stream_llm_tokens"}
        _assert_all_traces_valid()
