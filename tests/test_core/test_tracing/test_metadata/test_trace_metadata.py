from deepeval.tracing import observe, update_current_trace
from tests.test_core.test_tracing.conftest import trace_test


@observe()
def trace_with_metadata(data: str) -> str:
    update_current_trace(
        metadata={
            "user_id": "user_789",
            "request_id": "req_abc123",
            "source": "api",
        }
    )
    return f"Result: {data}"


@observe()
def trace_with_user_info(data: str) -> str:
    update_current_trace(
        user_id="user_123",
        thread_id="thread_456",
        metadata={
            "subscription_tier": "premium",
            "region": "us-west-2",
        },
    )
    return data


@observe()
def trace_with_full_context(query: str) -> str:
    update_current_trace(
        name="search_workflow",
        user_id="user_001",
        thread_id="conv_123",
        metadata={
            "workflow_type": "search",
            "version": "2.0",
            "features_enabled": ["semantic_search", "reranking"],
        },
    )
    return f"Searched: {query}"


@observe()
def outer_function(data: str) -> str:
    update_current_trace(metadata={"outer_key": "outer_value"})
    return inner_function(data)


@observe()
def inner_function(data: str) -> str:
    return f"Inner: {data}"


class TestTraceMetadata:

    @trace_test("metadata/trace_basic_metadata_schema.json")
    def test_basic_trace_metadata(self):
        trace_with_metadata("test")

    @trace_test("metadata/trace_user_info_schema.json")
    def test_trace_with_user_info(self):
        trace_with_user_info("data")

    @trace_test("metadata/trace_full_context_schema.json")
    def test_trace_full_context(self):
        trace_with_full_context("AI search")

    @trace_test("metadata/trace_nested_spans_schema.json")
    def test_trace_metadata_persists(self):
        outer_function("test")
