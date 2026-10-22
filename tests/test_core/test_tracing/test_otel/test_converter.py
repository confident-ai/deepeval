
"""``convert_span`` against real confident-trace spans and its receiver vectors.

``fixtures/`` holds unmodified copies of confident-trace's ``spec/`` vectors.
"""

import json
from pathlib import Path

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext

from deepeval.test_case import ToolCall
from deepeval.tracing.otel import adapters
from deepeval.tracing.otel.adapters import Adapter
from deepeval.tracing.otel.converter import (
    convert_span,
    is_ai_span,
    resolve_span_type,
)
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
    SpanType,
    ToolSpan,
    TraceSpanStatus,
)

FIXTURES = Path(__file__).parent / "fixtures"
GENAI_CASES = json.loads((FIXTURES / "genai-vectors.json").read_text())["cases"]
FIELD_CASES = json.loads((FIXTURES / "span-field-vectors.json").read_text())[
    "cases"
]
TOOL_FIELDS = ("tools_called", "expected_tools")


@pytest.fixture
def ct():
    module = pytest.importorskip("confident_trace")
    module.shutdown()
    yield module
    module.shutdown()


@pytest.fixture
def export(ct):
    """Start confident-trace; the result converts its exported spans by name."""

    def start(instrumentations=(), resource=None, **options):
        exporter = InMemorySpanExporter()
        provider = TracerProvider(
            resource=resource or Resource.create({}), shutdown_on_exit=False
        )
        ct.init(
            tracer_provider=provider,
            exporter=exporter,
            instrumentations=instrumentations,
            **options,
        )

        def converted():
            ct.flush()
            return {
                span.name: convert_span(span)
                for span in exporter.get_finished_spans()
            }

        return converted

    return start


def synthetic_span(attributes=None, events=(), name="span", scope=None):
    return ReadableSpan(
        name=name,
        context=SpanContext(trace_id=1, span_id=2, is_remote=False),
        attributes=attributes or {},
        events=[Event(event["name"], event["attributes"]) for event in events],
        start_time=1_000,
        end_time=2_000,
        instrumentation_scope=InstrumentationScope(scope) if scope else None,
    )


def expected_value(field, value):
    if field in TOOL_FIELDS:
        return [ToolCall.model_validate(call) for call in value]
    return value


################ Content fields ################


@pytest.mark.parametrize("case", FIELD_CASES, ids=lambda case: case["field"])
def test_span_field_round_trips(ct, export, case):
    converted = export()
    with ct.span("step"):
        ct.update_span(**{case["field"]: case["value"]})

    span = converted()["step"].span
    assert getattr(span, case["field"]) == expected_value(
        case["field"], case["value"]
    )


@pytest.mark.parametrize("case", FIELD_CASES, ids=lambda case: case["field"])
def test_trace_field_round_trips(ct, export, case):
    converted = export()
    with ct.span("root"):
        ct.update_trace(**{case["field"]: case["value"]})

    trace_fields = converted()["root"].trace_fields
    assert trace_fields[case["field"]] == expected_value(
        case["field"], case["value"]
    )


def test_json_text_stays_text(ct, export):
    converted = export()
    with ct.span("step"):
        ct.update_span(input='{"a": 1}', output="澳門 😀")

    span = converted()["step"].span
    assert span.input == '{"a": 1}'
    assert span.output == "澳門 😀"


def test_oversized_content_keeps_the_truncation_marker(ct, export):
    converted = export(max_content_bytes=64)
    with ct.span("step"):
        ct.update_span(input="a" * 100)

    assert converted()["step"].span.input == "[truncated]"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("[]", []),
        ('[{"bad": 1}]', None),
        ('[{"name": "a"}, {"bad": 1}]', [ToolCall(name="a")]),
        ('[{"name": "a", "type": "unknown"}]', [ToolCall(name="a")]),
        ("not json", None),
    ],
)
def test_tool_calls_keep_valid_entries(raw, expected):
    span = convert_span(
        synthetic_span({"confident.span.tools_called": raw})
    ).span
    assert span.tools_called == expected


################ Trace fields ################


def test_trace_fields_come_from_the_entry_span(ct, export):
    converted = export()
    with ct.span("root", thread_id="thread", user_id="user", tags=["a"]):
        ct.update_trace(
            input="hi",
            output={"answer": 1},
            metadata={"release": "v2"},
            test_case_id="case",
            turn_id="turn",
            metric_collection="collection",
            environment="staging",
        )
        with ct.span("child"):
            pass

    spans = converted()
    assert spans["child"].trace_fields == {}
    assert spans["root"].trace_fields == {
        "name": "root",
        "tags": ["a"],
        "metadata": {"release": "v2"},
        "thread_id": "thread",
        "user_id": "user",
        "environment": "staging",
        "test_case_id": "case",
        "turn_id": "turn",
        "metric_collection": "collection",
        "input": "hi",
        "output": {"answer": 1},
    }


def test_resource_environment_wins_and_unknown_values_are_dropped(ct, export):
    converted = export(
        resource=Resource.create({"confident.trace.environment": "testing"})
    )
    with ct.span("root"):
        ct.update_trace(environment="staging")
    assert converted()["root"].trace_fields["environment"] == "testing"

    span = synthetic_span({"confident.trace.environment": "prod"})
    assert "environment" not in convert_span(span).trace_fields


################ Span types ################


def test_application_span_types(ct, export):
    converted = export()

    @ct.span(type="tool", name="lookup")
    def lookup(city):
        return {"temp": 20}

    with ct.span("root"):
        lookup("Macau")
        with ct.span("planner", type="agent"):
            ct.update_span(input="plan", output="done")
        with ct.span("search", type="retriever"):
            ct.update_span(input="query", retrieval_context=["doc"])
        with ct.span(
            "generate",
            type="llm",
            model="gpt-4o",
            provider="openai",
            input_token_count=3,
            output_token_count=5,
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
        ):
            pass

    spans = {name: item.span for name, item in converted().items()}
    assert type(spans["root"]) is BaseSpan

    tool = spans["execute_tool lookup"]
    assert isinstance(tool, ToolSpan)
    assert tool.name == "lookup"
    assert tool.input == {"args": ["Macau"], "kwargs": {}}
    assert tool.output == {"temp": 20}

    agent = spans["planner"]
    assert isinstance(agent, AgentSpan)
    assert (agent.name, agent.input, agent.output) == (
        "planner",
        "plan",
        "done",
    )

    retriever = spans["search"]
    assert isinstance(retriever, RetrieverSpan)
    assert retriever.retrieval_context == ["doc"]
    assert retriever.embedder == ""

    llm = spans["generate"]
    assert isinstance(llm, LlmSpan)
    assert (llm.model, llm.provider) == ("gpt-4o", "OpenAI")
    assert (llm.input_token_count, llm.output_token_count) == (3, 5)
    assert (llm.cost_per_input_token, llm.cost_per_output_token) == (
        0.001,
        0.002,
    )


@pytest.mark.parametrize(
    "attributes, span_type, name",
    [
        (
            {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "a"},
            AgentSpan,
            "a",
        ),
        (
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "t"},
            ToolSpan,
            "t",
        ),
        ({"gen_ai.tool.name": "t"}, ToolSpan, "t"),
        ({"gen_ai.operation.name": "retrieval"}, RetrieverSpan, "span"),
        ({"gen_ai.operation.name": "chat"}, LlmSpan, "span"),
        # The platform keeps embeddings as base spans.
        ({"gen_ai.operation.name": "embeddings"}, BaseSpan, "span"),
        (
            {"confident.span.type": "custom", "gen_ai.operation.name": "chat"},
            LlmSpan,
            "span",
        ),
        (
            {"confident.span.type": "base", "gen_ai.operation.name": "chat"},
            BaseSpan,
            "span",
        ),
        ({}, BaseSpan, "span"),
    ],
)
def test_span_type_classification(attributes, span_type, name):
    span = convert_span(synthetic_span(attributes)).span
    assert type(span) is span_type
    assert span.name == name


def test_ids_parentage_and_errors(ct, export):
    converted = export()
    with pytest.raises(ValueError):
        with ct.span("root"):
            with ct.span("boom"):
                raise ValueError("secret")

    spans = converted()
    root, boom = spans["root"].span, spans["boom"].span
    assert root.parent_uuid is None
    assert boom.parent_uuid == root.uuid
    assert boom.trace_uuid == root.trace_uuid
    assert (len(boom.uuid), len(boom.trace_uuid)) == (16, 32)
    assert boom.status == TraceSpanStatus.ERRORED
    assert boom.error == "ValueError"
    assert boom.start_time <= boom.end_time


################ Model spans ################


@pytest.mark.parametrize("case", GENAI_CASES, ids=lambda case: case["name"])
def test_receiver_message_vectors(case):
    span = convert_span(
        synthetic_span(
            {"gen_ai.operation.name": "chat", **case["attributes"]},
            case["events"],
        )
    ).span
    assert span.input == case["expected"]["input"]
    # Output collapses to its text, which is what metrics score.
    texts = [message["content"] for message in case["expected"]["output"]]
    assert span.output == ("\n".join(texts) if texts else [])


def test_system_instructions_lead_the_input():
    span = convert_span(
        synthetic_span(
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.system_instructions": json.dumps(
                    [{"type": "text", "content": "be brief"}]
                ),
                "gen_ai.input.messages": json.dumps(
                    [
                        {
                            "role": "user",
                            "parts": [{"type": "text", "content": "q"}],
                        }
                    ]
                ),
            }
        )
    ).span
    assert span.input == [
        {"role": "System Instruction", "content": "be brief"},
        {"role": "user", "content": "q"},
    ]


@pytest.mark.parametrize(
    "attributes, provider",
    [
        ({"gen_ai.provider.name": "aws.bedrock"}, "Bedrock"),
        ({"gen_ai.provider.name": "gcp.gen_ai"}, "Gemini"),
        ({"gen_ai.provider.name": "gcp.vertex_ai"}, "VertexAI"),
        ({"gen_ai.provider.name": "azure.ai.openai"}, "Azure"),
        ({"gen_ai.provider.name": "anthropic"}, "Anthropic"),
        ({"gen_ai.system": "openai"}, "OpenAI"),
        ({"gen_ai.request.model": "claude-sonnet-4"}, "Anthropic"),
    ],
)
def test_provider_normalization(attributes, provider):
    span = convert_span(
        synthetic_span({"gen_ai.operation.name": "chat", **attributes})
    ).span
    assert span.provider == provider


def test_legacy_token_counts():
    span = convert_span(
        synthetic_span(
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.usage.prompt_tokens": 4,
                "gen_ai.usage.completion_tokens": 9,
            }
        )
    ).span
    assert (span.input_token_count, span.output_token_count) == (4, 9)


def test_openai_call_with_tool_calls(export, openai_client):
    converted = export(instrumentations=("openai",))
    openai_client("tool_call").chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "weather?"},
        ],
    )

    span = converted()["chat gpt-4o-mini"].span
    assert isinstance(span, LlmSpan)
    assert (span.model, span.provider, span.integration) == (
        "gpt-4o-mini",
        "OpenAI",
        "OpenAI",
    )
    assert (span.input_token_count, span.output_token_count) == (11, 7)
    assert span.input == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather?"},
    ]
    assert span.output == [
        {
            "role": "assistant",
            "content": {
                "type": "tool_call",
                "id": "t1",
                "name": "lookup",
                "arguments": {"city": "Macau"},
            },
        }
    ]
    assert span.tools_called == [
        ToolCall(name="lookup", input_parameters={"city": "Macau"})
    ]


def test_openai_call_with_text(export, openai_client):
    converted = export(instrumentations=("openai",))
    openai_client("text").chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "weather?"},
        ],
    )

    call = converted()["chat gpt-4o-mini"]
    assert call.span.output == "sunny"
    assert call.span.tools_called is None
    # As the trace's entry, the call also gives the trace its input and output.
    assert (call.trace_fields["input"], call.trace_fields["output"]) == (
        "weather?",
        "sunny",
    )


################ Rules every framework shares ################


def test_plain_text_tool_results_are_kept():
    span = convert_span(
        synthetic_span(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "lookup",
                "gen_ai.tool.call.arguments": '{"city": "Paris"}',
                "gen_ai.tool.call.result": "sunny",
            }
        )
    ).span
    assert (span.input, span.output) == ({"city": "Paris"}, "sunny")


def test_a_tool_reply_message_reads_as_its_response():
    reply = [
        {
            "role": "tool",
            "parts": [
                {"type": "tool_call_response", "id": "t1", "response": "sunny"}
            ],
        }
    ]
    span = convert_span(
        synthetic_span(
            {
                "gen_ai.operation.name": "execute_tool",
                "confident.span.output": json.dumps(reply),
            }
        )
    ).span
    assert span.output == "sunny"


def test_content_blocks_in_events_read_as_text():
    span = convert_span(
        synthetic_span(
            {"gen_ai.operation.name": "chat"},
            [
                {
                    "name": "gen_ai.user.message",
                    "attributes": {"content": '[{"text": "Weather?"}]'},
                },
                {
                    "name": "gen_ai.choice",
                    "attributes": {"message": '[{"text": "Sunny."}]'},
                },
            ],
        )
    ).span
    assert span.input == [{"role": "user", "content": "Weather?"}]
    assert span.output == "Sunny."


@pytest.mark.parametrize(
    "name, attributes, start_type",
    [
        ("invoke_agent planner", {}, SpanType.AGENT),
        ("execute_tool lookup", {}, SpanType.TOOL),
        ("chat", {}, SpanType.LLM),
        # A span that declares its own category isn't renamed into one.
        ("chat", {"confident.span.type": "custom"}, None),
        ("planner", {}, None),
    ],
)
def test_spans_still_opening_are_typed_by_their_name(
    name, attributes, start_type
):
    span = synthetic_span(attributes, name=name)
    assert resolve_span_type(span, at_start=True) == start_type
    assert resolve_span_type(span) is None


def test_retrieved_documents_become_retrieval_context():
    span = convert_span(
        synthetic_span(
            {
                "confident.span.type": "retriever",
                "confident.span.output": json.dumps(
                    [{"page_content": "a"}, {"text": "b", "score": 0.5}]
                ),
            }
        )
    ).span
    assert span.retrieval_context == ["a", "b"]


def test_conversation_id_and_latest_user_message():
    messages = [
        {"role": "user", "parts": [{"type": "text", "content": "first"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "ok"}]},
        {"role": "user", "parts": [{"type": "text", "content": "second"}]},
    ]
    converted = convert_span(
        synthetic_span(
            {
                "gen_ai.conversation.id": "thread",
                "confident.trace.input": json.dumps(messages),
            }
        )
    )
    assert converted.conversation_id == "thread"
    assert converted.trace_fields["input"] == "second"


class _FrameworkAdapter(Adapter):
    self_providers = frozenset({"framework"})

    def detect(self, span):
        return span.instrumentation_scope is not None

    def start_span_type(self, span):
        return "agent"

    def span_type(self, span):
        return "llm"

    def span_fields(self, span, span_type):
        return {"model": "claude-sonnet-4", "output": "ignored"}

    def trace_fields(self, span, span_type):
        return {"thread_id": "framework-thread"}

    def conversation_id(self, span):
        return "framework-conversation"


def test_adapters_fill_only_what_the_core_left_unset(monkeypatch):
    monkeypatch.setattr(adapters, "ADAPTERS", (_FrameworkAdapter(),))
    span = synthetic_span(
        {
            "gen_ai.provider.name": "framework",
            "confident.span.output": '"recorded"',
        },
        scope="framework",
    )

    assert is_ai_span(span)
    assert resolve_span_type(span, at_start=True) == SpanType.AGENT
    converted = convert_span(span)
    llm = converted.span
    assert isinstance(llm, LlmSpan)
    assert (llm.model, llm.output) == ("claude-sonnet-4", "recorded")
    # The framework's own name gives way to the model's real vendor.
    assert llm.provider == "Anthropic"
    assert converted.trace_fields == {"thread_id": "framework-thread"}
    assert converted.conversation_id == "framework-conversation"


################ Framework adapters ################


@pytest.mark.parametrize(
    "name, span_type",
    [
        ("VectorIndexRetriever.retrieve", RetrieverSpan),
        ("BaseRetriever.aretrieve", RetrieverSpan),
        ("VectorIndexRetriever._retrieve", BaseSpan),
        ("RetrieverQueryEngine.query", BaseSpan),
    ],
)
def test_llamaindex_retrievals_are_retriever_spans(name, span_type):
    documents = [{"text": "Paris is in France.", "score": 0.9}]
    span = synthetic_span(
        {
            "confident.span.integration": "LlamaIndex",
            "confident.span.output": json.dumps(documents),
        },
        name=name,
    )
    converted = convert_span(span).span
    assert type(converted) is span_type
    if span_type is RetrieverSpan:
        assert resolve_span_type(span, at_start=True) == SpanType.RETRIEVER
        assert converted.retrieval_context == ["Paris is in France."]


_PYDANTIC_AI_RUN = [
    {"role": "user", "parts": [{"type": "text", "content": "Weather?"}]},
    {
        "role": "assistant",
        "parts": [{"type": "tool_call", "name": "weather", "arguments": {}}],
    },
    {
        "role": "user",
        "parts": [{"type": "tool_call_response", "response": "sunny"}],
    },
    {"role": "assistant", "parts": [{"type": "text", "content": "Sunny."}]},
]


@pytest.mark.parametrize(
    "final_result, output", [("Sunny!", "Sunny!"), (None, "Sunny.")]
)
def test_pydantic_ai_agents_read_their_run(final_result, output):
    attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "weather_agent",
        "pydantic_ai.all_messages": json.dumps(_PYDANTIC_AI_RUN),
    }
    if final_result is not None:
        attributes["final_result"] = final_result
    span = synthetic_span(
        attributes, name="invoke_agent weather_agent", scope="pydantic-ai"
    )

    agent = convert_span(span).span

    assert isinstance(agent, AgentSpan)
    assert (agent.name, agent.input, agent.output) == (
        "weather_agent",
        "Weather?",
        output,
    )


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("result", ["sunny", "", False, 0])
def test_pydantic_ai_tool_attribute_formats(modern, result):
    arguments = "gen_ai.tool.call.arguments" if modern else "tool_arguments"
    response = "gen_ai.tool.call.result" if modern else "tool_response"
    span = synthetic_span(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "weather",
            arguments: '{"city":"Paris"}',
            response: result,
        },
        name="execute_tool weather",
        scope="pydantic-ai",
    )

    tool = convert_span(span).span

    assert isinstance(tool, ToolSpan)
    assert (tool.input, tool.output) == ({"city": "Paris"}, result)


@pytest.mark.parametrize(
    "kind, span_type",
    [
        ("LLM", LlmSpan),
        ("EMBEDDING", LlmSpan),
        ("AGENT", AgentSpan),
        ("TOOL", ToolSpan),
        ("RETRIEVER", RetrieverSpan),
        ("RERANKER", RetrieverSpan),
        ("CHAIN", BaseSpan),
        ("GUARDRAIL", BaseSpan),
    ],
)
def test_openinference_kinds(kind, span_type):
    span = synthetic_span(
        {"openinference.span.kind": kind, "tool.name": "step"}, name="step"
    )
    assert is_ai_span(span)
    assert type(convert_span(span).span) is span_type


def test_an_openinference_tool_that_names_no_tool_is_a_base_span():
    span = synthetic_span(
        {"openinference.span.kind": "TOOL"}, name="handoff to assistant"
    )
    assert resolve_span_type(span, at_start=True) == SpanType.TOOL
    assert type(convert_span(span).span) is BaseSpan


def test_openinference_model_calls():
    span = synthetic_span(
        {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-4o-mini",
            "llm.provider": "openai",
            "llm.token_count.prompt": 11,
            "llm.token_count.completion": 7,
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "Be brief.",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.contents.0.message_content.type": "text",
            "llm.input_messages.1.message.contents.0.message_content.text": "Weather?",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "t1",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "lookup",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city": "Paris"}',
            "session.id": "session",
            "user.id": "user",
            "tag.tags": ("weather",),
        },
        scope="openinference.instrumentation.openai",
    )

    converted = convert_span(span)
    llm = converted.span

    assert isinstance(llm, LlmSpan)
    assert (llm.model, llm.provider) == ("gpt-4o-mini", "OpenAI")
    assert (llm.input_token_count, llm.output_token_count) == (11, 7)
    assert llm.input == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather?"},
    ]
    assert llm.tools_called == [
        ToolCall(name="lookup", input_parameters={"city": "Paris"})
    ]
    assert (llm.integration, converted.trace_fields) == (
        "OpenInference",
        {"thread_id": "session", "user_id": "user", "tags": ["weather"]},
    )


def test_openinference_tools_and_retrievers():
    tool = convert_span(
        synthetic_span(
            {
                "openinference.span.kind": "TOOL",
                "tool.name": "lookup",
                "tool.description": "Looks up the weather.",
                "input.value": '{"city": "Paris"}',
                "input.mime_type": "application/json",
                "output.value": "sunny",
            },
            name="lookup",
        )
    ).span
    assert (tool.name, tool.description) == ("lookup", "Looks up the weather.")
    assert (tool.input, tool.output) == ({"city": "Paris"}, "sunny")

    retriever = convert_span(
        synthetic_span(
            {
                "openinference.span.kind": "RETRIEVER",
                "input.value": "Paris",
                "retrieval.documents.0.document.content": "Paris is in France.",
                "retrieval.documents.0.document.score": 0.9,
                "retrieval.documents.1.document.content": "It is sunny.",
            }
        )
    ).span
    assert retriever.input == "Paris"
    assert retriever.retrieval_context == [
        "Paris is in France.",
        "It is sunny.",
    ]


def test_content_confident_trace_could_not_serialize_is_missing():
    span = synthetic_span(
        {
            "gen_ai.operation.name": "invoke_agent",
            "confident.span.input": json.dumps("Weather?"),
            "confident.span.output": json.dumps({"response": "[unsupported]"}),
            "confident.trace.output": json.dumps(["[unsupported]"]),
        }
    )

    converted = convert_span(span)

    assert (converted.span.input, converted.span.output) == ("Weather?", None)
    assert "output" not in converted.trace_fields


@pytest.mark.parametrize(
    "recorded, arguments",
    [
        ({"args": [], "kwargs": {"city": "Paris"}}, {"city": "Paris"}),
        ({"kwargs": {"city": "Paris"}}, {"city": "Paris"}),
        ({"args": ["Paris"], "kwargs": {}}, {"args": ["Paris"], "kwargs": {}}),
        ({"city": "Paris"}, {"city": "Paris"}),
    ],
)
def test_a_tool_called_with_keywords_takes_them_as_input(recorded, arguments):
    span = synthetic_span(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "weather",
            "confident.span.input": json.dumps(recorded),
        }
    )
    assert convert_span(span).span.input == arguments


def test_a_turn_that_only_calls_tools_keeps_its_calls_as_output():
    output = [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": ""},
                {"type": "tool_call", "name": "lookup", "arguments": {}},
            ],
        }
    ]
    span = synthetic_span(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.output.messages": json.dumps(output),
        }
    )

    llm = convert_span(span).span

    assert llm.output == [
        {
            "role": "assistant",
            "content": {"type": "tool_call", "name": "lookup", "arguments": {}},
        }
    ]
    assert [call.name for call in llm.tools_called] == ["lookup"]


def test_a_conversation_is_answered_by_its_final_assistant_turn():
    def text(role, content):
        return {"role": role, "parts": [{"type": "text", "content": content}]}

    conversation = [
        text("user", "Weather?"),
        text("assistant", "Let me check."),
        text("tool", "sunny"),
        text("assistant", "Sunny."),
    ]
    span = synthetic_span({"confident.trace.output": json.dumps(conversation)})

    assert convert_span(span).trace_fields["output"] == "Sunny."
