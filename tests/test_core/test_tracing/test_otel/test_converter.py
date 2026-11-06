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
from opentelemetry.trace import SpanContext

from deepeval.test_case import ToolCall
from deepeval.tracing.otel.converter import convert_span
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
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


def synthetic_span(attributes=None, events=()):
    return ReadableSpan(
        name="span",
        context=SpanContext(trace_id=1, span_id=2, is_remote=False),
        attributes=attributes or {},
        events=[Event(event["name"], event["attributes"]) for event in events],
        start_time=1_000,
        end_time=2_000,
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
