import threading
import asyncio
import pytest

from deepeval.tracing import observe
from deepeval.tracing.context import current_span_context
from tests.test_core.test_tracing.conftest import trace_test


# ── Leaf helpers ──────────────────────────────────────────────────────────


@observe()
def stream_chunks(message: str):
    tokens = message.split()
    for token in tokens:
        yield {"type": "chunk", "data": {"token": token}}
    yield {"type": "final_response", "data": {"content": message}}
    return


@observe()
def stream_simple(data: str):
    for word in data.split():
        yield word


@observe()
def stream_with_error(data: str):
    yield "first"
    yield "second"
    if data == "error":
        raise ValueError("Stream error")
    yield "third"


@observe()
def process_item(item) -> dict:
    return {"processed": item}


@observe()
async def async_stream_chunks(message: str):
    tokens = message.split()
    for token in tokens:
        await asyncio.sleep(0.005)
        yield {"type": "chunk", "data": {"token": token}}
    yield {"type": "final_response", "data": {"content": message}}
    return


@observe()
async def async_stream_simple(data: str):
    for word in data.split():
        await asyncio.sleep(0.005)
        yield word


# ── Nested generators / observed functions ────────────────────────────────


@observe()
def outer_observe_consumes_inner_gen(message: str):
    """Regular @observe function that fully consumes an inner generator."""
    results = []
    for chunk in stream_chunks(message):
        results.append(chunk)
    return results


@observe()
def outer_observe_breaks_inner_gen(message: str):
    """Regular @observe function that breaks out of an inner generator."""
    for chunk in stream_chunks(message):
        if chunk["type"] == "final_response":
            return chunk["data"]
    return None


@observe()
def outer_gen_yields_from_inner_gen(message: str):
    """Outer generator that re-yields chunks from an inner generator."""
    yield {"type": "wrapper", "data": "start"}
    for chunk in stream_chunks(message):
        yield chunk
    yield {"type": "wrapper", "data": "end"}


@observe()
def outer_gen_breaks_inner_gen(message: str):
    """Outer generator that breaks the inner generator mid-stream."""
    yield {"type": "wrapper", "data": "start"}
    for chunk in stream_chunks(message):
        yield chunk
        if chunk["type"] == "final_response":
            break
    yield {"type": "wrapper", "data": "end"}


@observe()
def outer_gen_calls_regular_observe(data: str):
    """Generator that calls a regular @observe function between yields."""
    for word in data.split():
        enriched = process_item(word)
        yield enriched


@observe()
def three_level_gen(data: str):
    """Top-level generator → mid-level generator → leaf generator."""
    yield {"level": "top", "stage": "start"}
    for chunk in mid_level_gen(data):
        yield {"level": "top", "inner": chunk}
    yield {"level": "top", "stage": "end"}


@observe()
def mid_level_gen(data: str):
    """Mid-level generator that consumes a leaf generator."""
    yield {"level": "mid", "stage": "start"}
    for word in stream_simple(data):
        yield {"level": "mid", "word": word}
    yield {"level": "mid", "stage": "end"}


@observe()
def sibling_generators(data: str):
    """Observed function that consumes two sibling generators sequentially."""
    results_a = list(stream_simple(data))
    results_b = list(stream_chunks(data))
    return {"simple": results_a, "chunks": results_b}


@observe()
async def async_outer_observe_consumes_inner_gen(message: str):
    """Async regular function that fully consumes an async inner generator."""
    results = []
    async for chunk in async_stream_chunks(message):
        results.append(chunk)
    return results


@observe()
async def async_outer_gen_yields_from_inner_gen(message: str):
    """Async outer generator that re-yields from an async inner generator."""
    yield {"type": "wrapper", "data": "start"}
    async for chunk in async_stream_chunks(message):
        yield chunk
    yield {"type": "wrapper", "data": "end"}


@observe()
async def async_outer_gen_breaks_inner(message: str):
    """Async outer generator that breaks inner generator after final_response."""
    yield {"type": "wrapper", "data": "start"}
    async for chunk in async_stream_chunks(message):
        yield chunk
        if chunk["type"] == "final_response":
            break
    yield {"type": "wrapper", "data": "end"}


@observe()
async def async_three_level_gen(data: str):
    """Async three-level nesting: top → mid → leaf."""
    yield {"level": "top", "stage": "start"}
    async for chunk in async_mid_level_gen(data):
        yield {"level": "top", "inner": chunk}
    yield {"level": "top", "stage": "end"}


@observe()
async def async_mid_level_gen(data: str):
    yield {"level": "mid", "stage": "start"}
    async for word in async_stream_simple(data):
        yield {"level": "mid", "word": word}
    yield {"level": "mid", "stage": "end"}



class TestSyncGeneratorContextSafety:

    def test_thread_boundary(self):
        """Generator created in main thread, consumed in worker thread."""
        gen = stream_chunks("hello world")
        results = []
        error_holder = []

        def consume(g):
            try:
                for chunk in g:
                    results.append(chunk)
            except Exception as e:
                error_holder.append(e)

        t = threading.Thread(target=consume, args=(gen,))
        t.start()
        t.join()

        assert not error_holder, f"Worker thread raised: {error_holder[0]}"
        assert len(results) == 3
        assert results[-1]["type"] == "final_response"
        assert current_span_context.get() is None

    def test_interleaved_creation(self):
        """Two generators created back-to-back before either is consumed."""
        gen_a = stream_chunks("alpha beta")
        gen_b = stream_chunks("gamma delta")

        results_a = list(gen_a)
        results_b = list(gen_b)

        assert len(results_a) == 3
        assert len(results_b) == 3
        assert results_a[-1]["data"]["content"] == "alpha beta"
        assert results_b[-1]["data"]["content"] == "gamma delta"
        assert current_span_context.get() is None

    def test_run_in_executor(self):
        """Sync generator consumed via run_in_executor (FastAPI pattern)."""
        async def async_consumer():
            loop = asyncio.get_event_loop()
            gen = stream_chunks("one two three")

            def drain(g):
                return list(g)

            return await loop.run_in_executor(None, drain, gen)

        results = asyncio.get_event_loop().run_until_complete(async_consumer())

        assert len(results) == 4
        assert results[-1]["type"] == "final_response"

    def test_break_after_final_response(self):
        """Consumer breaks after receiving final_response (GeneratorExit)."""
        result = None
        for chunk in stream_chunks("hello world"):
            if chunk["type"] == "final_response":
                result = chunk["data"]
                break

        assert result is not None
        assert result["content"] == "hello world"
        assert current_span_context.get() is None

    def test_next_then_abandon(self):
        """Consumer calls next() once then drops the generator."""
        gen = stream_simple("alpha beta gamma")
        first = next(gen)
        assert first == "alpha"
        del gen
        assert current_span_context.get() is None

    @trace_test("generators/context_safety_sync_observe_between_yields_schema.json")
    def test_consumer_calls_observe_between_yields(self):
        """Consumer calls another @observe function between generator yields."""
        gen = stream_simple("one two three")
        processed = []
        for word in gen:
            result = process_item(word)
            processed.append(result)

        assert len(processed) == 3
        assert processed[0] == {"processed": "one"}
        assert current_span_context.get() is None

    def test_error_still_propagates(self):
        """Exceptions inside the generator still propagate correctly."""
        collected = []
        with pytest.raises(ValueError, match="Stream error"):
            for token in stream_with_error("error"):
                collected.append(token)

        assert collected == ["first", "second"]
        assert current_span_context.get() is None

    @trace_test("generators/context_safety_sync_full_consumption_schema.json")
    def test_full_consumption_still_works(self):
        """Normal full consumption (yield then return) still works."""
        results = list(stream_chunks("a b c"))

        assert len(results) == 4
        assert results[-1]["type"] == "final_response"
        assert current_span_context.get() is None

    def test_sequential_calls_no_context_leak(self):
        """Multiple sequential calls don't leak context across calls."""
        for chunk in stream_chunks("first call"):
            if chunk["type"] == "final_response":
                break

        results = list(stream_chunks("second call"))
        assert results[-1]["data"]["content"] == "second call"
        assert current_span_context.get() is None

    def test_many_interleaved_generators(self):
        """Stress test: many generators created before any are consumed."""
        gens = [stream_simple(f"gen {i}") for i in range(10)]
        all_results = [list(g) for g in gens]

        assert len(all_results) == 10
        for i, results in enumerate(all_results):
            assert results == ["gen", str(i)]
        assert current_span_context.get() is None


# ── Nested sync tests ────────────────────────────────────────────────────


class TestSyncNestedGeneratorContext:

    @trace_test("generators/context_safety_sync_nested_consume_schema.json")
    def test_observe_consumes_inner_gen(self):
        """Regular @observe function fully consuming an inner generator."""
        results = outer_observe_consumes_inner_gen("hello world")

        assert len(results) == 3
        assert results[-1]["type"] == "final_response"
        assert current_span_context.get() is None

    def test_observe_breaks_inner_gen(self):
        """Regular @observe function breaking out of an inner generator early."""
        result = outer_observe_breaks_inner_gen("hello world")

        assert result is not None
        assert result["content"] == "hello world"
        assert current_span_context.get() is None

    @trace_test("generators/context_safety_sync_gen_yields_inner_schema.json")
    def test_gen_yields_from_inner_gen(self):
        """Outer generator re-yielding all chunks from an inner generator."""
        results = list(outer_gen_yields_from_inner_gen("alpha beta"))

        assert results[0] == {"type": "wrapper", "data": "start"}
        assert results[-1] == {"type": "wrapper", "data": "end"}
        inner_chunks = [r for r in results if r.get("type") == "chunk"]
        assert len(inner_chunks) == 2
        assert current_span_context.get() is None

    def test_gen_breaks_inner_gen(self):
        """Outer generator breaking inner generator after final_response."""
        results = list(outer_gen_breaks_inner_gen("x y"))

        assert results[0] == {"type": "wrapper", "data": "start"}
        assert results[-1] == {"type": "wrapper", "data": "end"}
        final = [r for r in results if r.get("type") == "final_response"]
        assert len(final) == 1
        assert current_span_context.get() is None

    @trace_test("generators/context_safety_sync_gen_observe_between_schema.json")
    def test_gen_calls_regular_observe_between_yields(self):
        """Generator calling a regular @observe function between every yield."""
        results = list(outer_gen_calls_regular_observe("a b c"))

        assert len(results) == 3
        assert results[0] == {"processed": "a"}
        assert results[2] == {"processed": "c"}
        assert current_span_context.get() is None

    @trace_test("generators/context_safety_sync_three_level_schema.json")
    def test_three_level_nesting(self):
        """Three levels deep: top gen → mid gen → leaf gen."""
        results = list(three_level_gen("x y"))

        assert results[0] == {"level": "top", "stage": "start"}
        assert results[-1] == {"level": "top", "stage": "end"}
        mid_words = [
            r["inner"]["word"]
            for r in results
            if isinstance(r.get("inner"), dict) and "word" in r["inner"]
        ]
        assert mid_words == ["x", "y"]
        assert current_span_context.get() is None

    @trace_test("generators/context_safety_sync_siblings_schema.json")
    def test_sibling_generators(self):
        """Observed function consuming two sibling generators sequentially."""
        result = sibling_generators("a b")

        assert result["simple"] == ["a", "b"]
        assert len(result["chunks"]) == 3
        assert result["chunks"][-1]["type"] == "final_response"
        assert current_span_context.get() is None

    def test_nested_gen_with_break_at_every_level(self):
        """Consumer breaks the outer gen, which itself breaks the inner gen."""
        collected = []
        for chunk in outer_gen_breaks_inner_gen("one two three"):
            collected.append(chunk)
            if chunk.get("type") == "final_response":
                break

        assert any(c.get("type") == "final_response" for c in collected)
        assert current_span_context.get() is None

    def test_nested_gen_inner_error_propagates(self):
        """Error in inner generator propagates cleanly through outer observe."""
        @observe()
        def outer_with_erroring_inner():
            results = []
            for token in stream_with_error("error"):
                results.append(token)
            return results

        with pytest.raises(ValueError, match="Stream error"):
            outer_with_erroring_inner()

        assert current_span_context.get() is None


# ── Async tests ───────────────────────────────────────────────────────────


class TestAsyncGeneratorContextSafety:

    @trace_test("generators/context_safety_async_full_consumption_schema.json")
    @pytest.mark.asyncio
    async def test_async_full_consumption(self):
        """Normal async generator full consumption."""
        results = []
        async for chunk in async_stream_chunks("hello world"):
            results.append(chunk)

        assert len(results) == 3
        assert results[-1]["type"] == "final_response"
        assert current_span_context.get() is None

    @pytest.mark.asyncio
    async def test_async_break_after_final_response(self):
        """Async consumer breaks after final_response."""
        result = None
        async for chunk in async_stream_chunks("hello world"):
            if chunk["type"] == "final_response":
                result = chunk["data"]
                break

        assert result is not None
        assert result["content"] == "hello world"
        assert current_span_context.get() is None

    @pytest.mark.asyncio
    async def test_async_interleaved_creation(self):
        """Two async generators created before either is consumed."""
        gen_a = async_stream_simple("alpha beta")
        gen_b = async_stream_simple("gamma delta")

        results_a = [item async for item in gen_a]
        results_b = [item async for item in gen_b]

        assert results_a == ["alpha", "beta"]
        assert results_b == ["gamma", "delta"]
        assert current_span_context.get() is None

    @pytest.mark.asyncio
    async def test_async_sequential_no_leak(self):
        """Sequential async generator calls don't leak context."""
        async for chunk in async_stream_chunks("call one"):
            if chunk["type"] == "final_response":
                break

        results = []
        async for chunk in async_stream_chunks("call two"):
            results.append(chunk)

        assert results[-1]["data"]["content"] == "call two"
        assert current_span_context.get() is None


# ── Nested async tests ───────────────────────────────────────────────────


class TestAsyncNestedGeneratorContext:

    @trace_test("generators/context_safety_async_nested_consume_schema.json")
    @pytest.mark.asyncio
    async def test_async_observe_consumes_inner_gen(self):
        """Async regular function fully consuming an async inner generator."""
        results = await async_outer_observe_consumes_inner_gen("hello world")

        assert len(results) == 3
        assert results[-1]["type"] == "final_response"
        assert current_span_context.get() is None

    @trace_test("generators/context_safety_async_gen_yields_inner_schema.json")
    @pytest.mark.asyncio
    async def test_async_gen_yields_from_inner_gen(self):
        """Async outer generator re-yielding from an async inner generator."""
        results = []
        async for chunk in async_outer_gen_yields_from_inner_gen("alpha beta"):
            results.append(chunk)

        assert results[0] == {"type": "wrapper", "data": "start"}
        assert results[-1] == {"type": "wrapper", "data": "end"}
        inner_chunks = [r for r in results if r.get("type") == "chunk"]
        assert len(inner_chunks) == 2
        assert current_span_context.get() is None

    @pytest.mark.asyncio
    async def test_async_gen_breaks_inner(self):
        """Async outer generator breaking inner after final_response."""
        results = []
        async for chunk in async_outer_gen_breaks_inner("x y"):
            results.append(chunk)

        assert results[0] == {"type": "wrapper", "data": "start"}
        assert results[-1] == {"type": "wrapper", "data": "end"}
        final = [r for r in results if r.get("type") == "final_response"]
        assert len(final) == 1
        assert current_span_context.get() is None

    @trace_test("generators/context_safety_async_three_level_schema.json")
    @pytest.mark.asyncio
    async def test_async_three_level_nesting(self):
        """Async three levels deep: top gen → mid gen → leaf gen."""
        results = []
        async for chunk in async_three_level_gen("x y"):
            results.append(chunk)

        assert results[0] == {"level": "top", "stage": "start"}
        assert results[-1] == {"level": "top", "stage": "end"}
        mid_words = [
            r["inner"]["word"]
            for r in results
            if isinstance(r.get("inner"), dict) and "word" in r["inner"]
        ]
        assert mid_words == ["x", "y"]
        assert current_span_context.get() is None

    @pytest.mark.asyncio
    async def test_async_nested_break_at_every_level(self):
        """Consumer breaks outer async gen, which breaks inner async gen."""
        collected = []
        async for chunk in async_outer_gen_breaks_inner("one two three"):
            collected.append(chunk)
            if chunk.get("type") == "final_response":
                break

        assert any(c.get("type") == "final_response" for c in collected)
        assert current_span_context.get() is None
