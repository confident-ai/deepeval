import pytest
from deepeval.tracing import observe, update_llm_span
from tests.test_core.test_tracing.conftest import trace_test


@observe(type="llm", model="gpt-4")
def streaming_llm(prompt: str):
    tokens = ["Hello", " ", "world", "!"]
    for token in tokens:
        yield token
    update_llm_span(
        input_token_count=len(prompt.split()),
        output_token_count=len(tokens),
    )


@observe()
def streaming_processor(data: str):
    chunks = data.split()
    for chunk in chunks:
        yield f"[{chunk}]"


@observe()
def streaming_with_nested_call(data: str):
    yield "Start"
    result = non_streaming_helper(data)
    yield result
    yield "End"


@observe()
def non_streaming_helper(data: str) -> str:
    return f"Processed: {data}"


@observe(type="llm", model="streaming-model")
def streaming_with_updates(prompt: str):
    tokens = prompt.split()
    total_tokens = 0
    for token in tokens:
        yield token
        total_tokens += 1
    update_llm_span(
        input_token_count=len(prompt.split()),
        output_token_count=total_tokens,
    )


@observe()
def streaming_with_error(data: str):
    yield "First"
    yield "Second"
    if data == "error":
        raise ValueError("Simulated error")
    yield "Third"


class TestSyncGenerator:

    @trace_test("generators/sync_streaming_llm_schema.json")
    def test_streaming_llm(self):
        list(streaming_llm("Test prompt"))

    @trace_test("generators/sync_streaming_processor_schema.json")
    def test_streaming_processor(self):
        list(streaming_processor("one two three"))

    @trace_test("generators/sync_streaming_nested_schema.json")
    def test_streaming_with_nested(self):
        list(streaming_with_nested_call("test"))

    @trace_test("generators/sync_streaming_updates_schema.json")
    def test_streaming_with_updates(self):
        list(streaming_with_updates("one two three four"))

    def test_streaming_error_handling(self):
        gen = streaming_with_error("error")
        results = []
        with pytest.raises(ValueError, match="Simulated error"):
            for token in gen:
                results.append(token)
        assert results == ["First", "Second"]
