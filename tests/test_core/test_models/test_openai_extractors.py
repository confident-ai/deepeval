from deepeval.openai.extractors import (
    extract_input_parameters_from_completion,
    safe_extract_input_parameters,
)


def test_extract_input_parameters_stringifies_multimodal_user_content_list():
    # simulate OpenAI chat payload where the user content is a list of parts
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What fruit is shown?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/banana.jpg"},
                    },
                ],
            },
        ],
    }

    params = extract_input_parameters_from_completion(kwargs)

    # ensure the params input is a string and not a list of parts
    # or a validation error will result
    # but check that content from the list is retained in the string
    assert isinstance(params.input, str)
    assert "What fruit is shown?" in params.input
    assert "banana.jpg" in params.input or "image" in params.input


def test_extract_input_parameters_with_assistant_message_without_content():
    # assistant messages that only carry tool_calls may omit `content`
    # entirely, which the OpenAI SDK request types allow
    kwargs = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "ping the server"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "ping", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "pong"},
        ],
    }

    params = extract_input_parameters_from_completion(kwargs)

    assert params.model == "gpt-4o"
    assert params.input == "ping the server"
    assert params.messages == [
        {"role": "user", "content": "ping the server"},
        {
            "id": "call_1",
            "call_id": "call_1",
            "name": "ping",
            "type": "function",
            "arguments": {},
        },
        {"call_id": "call_1", "type": "tool", "output": "pong"},
    ]


def test_safe_extract_input_parameters_with_assistant_message_without_content():
    # a missing `content` key on an assistant message must not degrade
    # the whole payload to InputParameters(model="NA")
    kwargs = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "ping the server"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "ping", "arguments": "{}"},
                    }
                ],
            },
        ],
    }

    params = safe_extract_input_parameters(True, kwargs)

    assert params.model == "gpt-4o"
    assert params.input == "ping the server"
