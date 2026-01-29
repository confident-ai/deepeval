from deepeval.tracing import observe, update_current_span
from tests.test_core.test_tracing.conftest import trace_test


@observe()
def span_with_metadata(data: str) -> str:
    update_current_span(
        metadata={
            "user_id": "user_123",
            "session_id": "sess_456",
            "environment": "production",
        }
    )
    return f"Processed: {data}"


@observe()
def span_with_complex_metadata(data: str) -> str:
    update_current_span(
        metadata={
            "request": {
                "method": "POST",
                "path": "/api/process",
            },
            "config": {
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            "tags": ["production", "v2"],
            "count": 42,
        }
    )
    return data


@observe(type="llm", model="gpt-4")
def llm_with_metadata(prompt: str) -> str:
    update_current_span(
        metadata={
            "model_version": "gpt-4-0125-preview",
            "system_prompt_hash": "abc123",
        }
    )
    return f"Response: {prompt}"


@observe(type="agent")
def agent_with_metadata(query: str) -> str:
    update_current_span(
        metadata={
            "execution_mode": "sequential",
            "retry_count": 0,
            "timeout_ms": 30000,
        }
    )
    return f"Agent: {query}"


class TestSpanMetadata:

    @trace_test("metadata/span_basic_metadata_schema.json")
    def test_basic_metadata(self):
        span_with_metadata("test")

    @trace_test("metadata/span_complex_metadata_schema.json")
    def test_complex_metadata(self):
        span_with_complex_metadata("data")

    @trace_test("metadata/llm_with_metadata_schema.json")
    def test_llm_with_metadata(self):
        llm_with_metadata("Hello")

    @trace_test("metadata/agent_with_metadata_schema.json")
    def test_agent_with_metadata(self):
        agent_with_metadata("query")
