"""v1 pydantic-ai spans, whose messages live in the `events` attribute.

The fixtures below are the literal shape pydantic-ai 0.8.1 writes with its
default instrumentation (`version=1`, `event_mode="attributes"`). Each entry
is `InstrumentedModel.event_to_dict(event)`, i.e. `{**event.body,
**event.attributes}`: the message body (`role`, `content`, `tool_calls`, ...)
plus the `gen_ai.system` / `gen_ai.message.index` attributes the v1 writers
add and the OTel event name, which `opentelemetry._events.Event.__init__`
seeds into `attributes` as `event.name`. The body shapes are those of
`UserPromptPart.otel_event`, `ModelResponse.otel_events` and
`ToolReturnPart.otel_event` -- note that a lone text part collapses to its
bare string.

These are pinned by hand on purpose: they have to keep working on pydantic-ai
versions where `version=1` no longer exists at all.
"""

import json
from types import SimpleNamespace

from deepeval.tracing.otel.attributes import ConfidentAttr
from deepeval.tracing.otel.utils import (
    _events_to_normalized_messages,
    _v1_events_to_llm_input_output,
    check_llm_input_from_gen_ai_attributes,
    check_pydantic_ai_agent_input_output,
    check_pydantic_ai_tools_called,
)

SYSTEM = "openai:gpt-4o"
USER_TEXT = "What's the weather in Paris?"
FINAL_TEXT = "It is sunny in Paris."
TOOL_ARGS = '{"city": "Paris"}'

SYSTEM_BODY = {"role": "system", "content": "You are a weather assistant."}
USER_BODY = {"role": "user", "content": USER_TEXT}
TOOL_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": TOOL_ARGS},
    }
]
# `ModelResponse.otel_events` collapses a lone text part to its bare string.
ASSISTANT_TOOL_CALL_BODY = {
    "role": "assistant",
    "content": "Let me check.",
    "tool_calls": TOOL_CALLS,
}
TOOL_BODY = {
    "role": "tool",
    "content": "sunny",
    "id": "call_1",
    "name": "get_weather",
}
FINAL_BODY = {"role": "assistant", "content": FINAL_TEXT}


def _event(body: dict, index: int, name: str) -> dict:
    """`event_to_dict` for an agent-span entry (no `gen_ai.system` there)."""
    return {
        **body,
        "gen_ai.message.index": index,
        "event.name": name,
    }


def _model_event(body: dict, index: int, name: str) -> dict:
    """`event_to_dict` for a model-span entry; `handle_messages` adds the
    system attribute on top of `messages_to_otel_events`."""
    return {
        **body,
        "gen_ai.system": SYSTEM,
        "gen_ai.message.index": index,
        "event.name": name,
    }


# `Agent._run_span_end_attributes` -> `all_messages_events`: the whole run
# history, with no `gen_ai.choice` wrapper. `gen_ai.message.index` counts
# messages (requests and responses alike).
AGENT_SPAN_EVENTS = [
    _event(SYSTEM_BODY, 0, "gen_ai.system.message"),
    _event(USER_BODY, 0, "gen_ai.user.message"),
    _event(ASSISTANT_TOOL_CALL_BODY, 1, "gen_ai.assistant.message"),
    _event(TOOL_BODY, 2, "gen_ai.tool.message"),
    _event(FINAL_BODY, 3, "gen_ai.assistant.message"),
]

# `InstrumentedModel.handle_messages` -> `events`: the messages this model
# request saw, then a `gen_ai.choice` entry wrapping the response. The two
# parts of one request share its index, and the choice carries no index.
MODEL_SPAN_EVENTS = [
    _model_event(SYSTEM_BODY, 0, "gen_ai.system.message"),
    _model_event(USER_BODY, 0, "gen_ai.user.message"),
    {
        "index": 0,
        "message": ASSISTANT_TOOL_CALL_BODY,
        "gen_ai.system": SYSTEM,
        "event.name": "gen_ai.choice",
    },
]

EXPECTED_MESSAGES = [
    {
        "role": "system",
        "parts": [{"type": "text", "content": "You are a weather assistant."}],
    },
    {"role": "user", "parts": [{"type": "text", "content": USER_TEXT}]},
    {
        "role": "assistant",
        "parts": [
            {"type": "text", "content": "Let me check."},
            {
                "type": "tool_call",
                "id": "call_1",
                "name": "get_weather",
                "arguments": TOOL_ARGS,
            },
        ],
    },
    {
        # v2 roles a tool result `user` (a `ToolReturnPart` can only live in a
        # `ModelRequest`), so the v1 normalizer must too.
        "role": "user",
        "parts": [
            {
                "type": "tool_call_response",
                "id": "call_1",
                "name": "get_weather",
                "result": "sunny",
            }
        ],
    },
    {
        "role": "assistant",
        "parts": [{"type": "text", "content": FINAL_TEXT}],
    },
]


def _agent_span() -> SimpleNamespace:
    return SimpleNamespace(
        attributes={
            "all_messages_events": json.dumps(AGENT_SPAN_EVENTS),
            ConfidentAttr.SPAN_TYPE: "agent",
        },
        parent=None,
    )


def _model_span() -> SimpleNamespace:
    return SimpleNamespace(
        attributes={
            "events": json.dumps(MODEL_SPAN_EVENTS),
            "gen_ai.operation.name": "chat",
            # Every pydantic-ai model span carries this, whatever the
            # instrumentation version.
            "model_request_parameters": json.dumps(
                {"output_tools": [], "allow_text_output": True}
            ),
            ConfidentAttr.SPAN_TYPE: "llm",
        },
        parent=None,
    )


def test_v1_events_normalize_to_v2_messages():
    assert (
        _events_to_normalized_messages(AGENT_SPAN_EVENTS) == EXPECTED_MESSAGES
    )


def test_v1_choice_wrapper_is_unwrapped_as_the_response():
    # The last model-span entry holds the response; it has no `role`.
    normalized = _events_to_normalized_messages(MODEL_SPAN_EVENTS)
    assert [m["role"] for m in normalized] == ["system", "user", "assistant"]
    assert normalized[-1]["parts"] == [
        {"type": "text", "content": "Let me check."},
        {
            "type": "tool_call",
            "id": "call_1",
            "name": "get_weather",
            "arguments": TOOL_ARGS,
        },
    ]


def test_v1_tool_event_name_is_not_mistaken_for_an_event_name():
    # `gen_ai.tool.message` bodies carry `name` for the tool, and it sits
    # next to the real `event.name`; reading `name` first ("get_weather")
    # matches no event type and drops the entry.
    normalized = _events_to_normalized_messages([AGENT_SPAN_EVENTS[3]])
    assert normalized == [EXPECTED_MESSAGES[3]]


def test_v1_tool_result_uses_the_v2_user_role():
    # Regression: v2 spells a tool result {"role": "user", "parts":
    # [{"type": "tool_call_response", ...}]}; a `tool` role here would make
    # the two instrumentation versions disagree for consumers that branch on
    # roles.
    normalized = _events_to_normalized_messages([AGENT_SPAN_EVENTS[3]])
    assert normalized[0]["role"] == "user"
    assert normalized[0]["parts"][0]["type"] == "tool_call_response"


def test_v1_message_roles_are_dispatchable_without_an_event_name():
    # An exporter that flattens the body and loses `event.name` still works:
    # the `role` in the body identifies the message.
    assert _events_to_normalized_messages([SYSTEM_BODY, USER_BODY]) == (
        EXPECTED_MESSAGES[:2]
    )


def test_v1_tool_calls_are_paired_with_their_results():
    calls = check_pydantic_ai_tools_called(_agent_span())
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].input_parameters == {"city": "Paris"}
    assert calls[0].output == "sunny"


def test_v1_agent_span_input_and_output():
    inputs, result = check_pydantic_ai_agent_input_output(_agent_span())
    assert inputs == [{"role": "user", "content": USER_TEXT}]
    assert result == {"role": "assistant", "content": FINAL_TEXT}


def test_v1_model_span_input_and_output_survive_model_request_parameters():
    # Regression: `model_request_parameters` is appended to the LLM input by
    # every pydantic-ai model span, so an empty-input guard that looks at the
    # accumulated `input` never fires and the events fallback is skipped.
    input, output = check_llm_input_from_gen_ai_attributes(_model_span())
    assert [m["content"] for m in input if m["content"] == USER_TEXT] == [
        USER_TEXT
    ]
    assert output == [
        {"role": "assistant", "content": "Let me check."},
        {
            "role": "assistant",
            "content": {
                "type": "tool_call",
                "id": "call_1",
                "name": "get_weather",
                "arguments": TOOL_ARGS,
            },
        },
    ]


def test_v1_agent_span_llm_fallback_reads_all_messages_events():
    # `all_messages_events` is the agent-span counterpart of `events`.
    span = _agent_span()
    del span.attributes[ConfidentAttr.SPAN_TYPE]
    input, output = check_llm_input_from_gen_ai_attributes(span)
    assert {"role": "user", "content": USER_TEXT} in input
    assert output is None


def _agent_span_attrs() -> SimpleNamespace:
    return SimpleNamespace(
        attributes={"all_messages_events": json.dumps(AGENT_SPAN_EVENTS)},
        parent=None,
    )


def test_v1_fallback_does_not_drop_the_last_message():
    # The fallback used to `pop()` the last entry unconditionally before
    # checking whether it was a `gen_ai.choice`, so a list that carries the
    # final response as an ordinary assistant message lost it.
    input, output = _v1_events_to_llm_input_output(_agent_span_attrs())
    assert output is None
    assert "It is sunny in Paris." in json.dumps(input)


def test_v1_fallback_returns_flattened_messages_not_raw_envelopes():
    # The fallback used to return `json.loads(events)` verbatim, leaving raw
    # event bodies (`gen_ai.message.index`, `event.name`, ...) where every
    # other path hands over `{role, content}`.
    input, _ = _v1_events_to_llm_input_output(_agent_span_attrs())
    assert all(set(m) == {"role", "content"} for m in input)
    assert all("event.name" not in m for m in input)


def test_nested_event_form_is_still_unwrapped():
    # `messages_to_otel_events` yields {"name", "body"} in memory before
    # `event_to_dict` flattens it; keep accepting that shape.
    nested = [
        {"name": "gen_ai.user.message", "body": dict(USER_BODY)},
        {
            "name": "gen_ai.tool.message",
            "body": {
                "content": "sunny",
                "role": "tool",
                "id": "c",
                "name": "w",
            },
        },
    ]
    assert _events_to_normalized_messages(nested) == [
        {"role": "user", "parts": [{"type": "text", "content": USER_TEXT}]},
        {
            "role": "user",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "c",
                    "name": "w",
                    "result": "sunny",
                }
            ],
        },
    ]
