from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.trace.status import Status, StatusCode


# Create a simple span context
span_context = SpanContext(
    trace_id=1,  # Simple trace ID
    span_id=1,  # Simple span ID
    is_remote=False,
    trace_flags=TraceFlags(0x01),  # Sampled flag
)

# Create the ReadableSpan with one attribute
readable_span = ReadableSpan(
    name="test_span",
    context=span_context,
    attributes={
        "agent_name": "test_agent",
        "model_name": "gpt-4",
        "logfire.msg": "test_agent run",
        "confident.span.name": "test_agent",
        "confident.span.type": "agent",
        "logfire.json_schema": '{"type": "object", "properties": {"pydantic_ai.all_messages": {"type": "array"}, "gen_ai.system_instructions": {"type": "array"}, "final_result": {"type": "object"}}}',
        "confident.trace.name": "test_trace_name",
        "confident.trace.tags": '["test_tag", "source:test"]',
        "confident.span.prompt": '{"alias": "test_agent", "version": "00.00.01"}',
        "confident.trace.metadata": '{"prompt_version": "00.00.01"}',
        "pydantic_ai.all_messages": """[
    {
        "role": "user",
        "parts": [
            {
                "type": "text",
                "content": "What should I do next?"
            }
        ]
    },
    {
        "role": "assistant",
        "parts": [
            {
                "type": "thinking",
                "content": "Test thinking part 1"
            },
            {
                "type": "thinking",
                "content": "Test thinking part 2"
            },
            {
                "type": "tool_call",
                "id": "call_test123",
                "name": "test_tool",
                "arguments": "{\\"query\\": \\"test query\\"}"
            }
        ],
        "finish_reason": "stop"
    },
    {
        "role": "user",
        "parts": [
            {
                "type": "tool_call_response",
                "id": "call_test123",
                "name": "test_tool",
                "result": "Test tool result"
            }
        ]
    },
    {
        "role": "assistant",
        "parts": [
            {
                "type": "thinking",
                "content": "Test final thinking"
            },
            {
                "type": "text",
                "content": "Final response text"
            }
        ],
        "finish_reason": "stop"
    }
]""",
        "confident.trace.thread_id": "test_thread_id",
        "gen_ai.usage.input_tokens": 1000,
        "gen_ai.system_instructions": '[{"type": "text", "content": "You are a test assistant. Follow these instructions: 1. Be concise 2. Use tools when needed 3. Provide clear responses"}]',
        "gen_ai.usage.output_tokens": 500,
        "confident.trace.environment": "development",
        "gen_ai.usage.details.reasoning_tokens": 300,
        "gen_ai.operation.name": "chat",
        "model_request_parameters": '{"temperature": 0.7, "max_tokens": 2048, "top_p": 0.9, "frequency_penalty": 0.5, "presence_penalty": 0.2}',
    },  # Single attribute
    status=Status(StatusCode.OK),
    start_time=1000000000,  # nanoseconds since epoch
    end_time=1000001000,  # nanoseconds since epoch
)

list_of_readable_spans = [readable_span]

llm_readable_span = ReadableSpan(
    name="test_span",
    context=span_context,
    attributes={
        "model_name": "gpt-4",
        "logfire.msg": "test_agent run",
        "confident.span.name": "test_agent",
        "logfire.json_schema": '{"type": "object", "properties": {"pydantic_ai.all_messages": {"type": "array"}, "gen_ai.system_instructions": {"type": "array"}, "final_result": {"type": "object"}}}',
        "confident.trace.name": "test_trace_name",
        "confident.trace.tags": '["test_tag", "source:test"]',
        "confident.span.prompt": '{"alias": "test_agent", "version": "00.00.01"}',
        "confident.trace.metadata": '{"prompt_version": "00.00.01"}',
        "pydantic_ai.all_messages": """[
    {
        "role": "user",
        "parts": [
            {
                "type": "text",
                "content": "What should I do next?"
            }
        ]
    },
    {
        "role": "assistant",
        "parts": [
            {
                "type": "thinking",
                "content": "Test thinking part 1"
            },
            {
                "type": "thinking",
                "content": "Test thinking part 2"
            },
            {
                "type": "tool_call",
                "id": "call_test123",
                "name": "test_tool",
                "arguments": "{\\"query\\": \\"test query\\"}"
            }
        ],
        "finish_reason": "stop"
    },
    {
        "role": "user",
        "parts": [
            {
                "type": "tool_call_response",
                "id": "call_test123",
                "name": "test_tool",
                "result": "Test tool result"
            }
        ]
    },
    {
        "role": "assistant",
        "parts": [
            {
                "type": "thinking",
                "content": "Test final thinking"
            },
            {
                "type": "text",
                "content": "Final response text"
            }
        ],
        "finish_reason": "stop"
    }
]""",
        "confident.trace.thread_id": "test_thread_id",
        "gen_ai.usage.input_tokens": 1000,
        "gen_ai.system_instructions": '[{"type": "text", "content": "You are a test assistant. Follow these instructions: 1. Be concise 2. Use tools when needed 3. Provide clear responses"}]',
        "gen_ai.usage.output_tokens": 500,
        "confident.trace.environment": "development",
        "gen_ai.usage.details.reasoning_tokens": 300,
        "gen_ai.operation.name": "chat",
        "model_request_parameters": '{"temperature": 0.7, "max_tokens": 2048, "top_p": 0.9, "frequency_penalty": 0.5, "presence_penalty": 0.2}',
    },  # Single attribute
    status=Status(StatusCode.OK),
    start_time=1000000000,  # nanoseconds since epoch
    end_time=1000001000,  # nanoseconds since epoch
)

llm_span_list = [llm_readable_span]
