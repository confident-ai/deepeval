import pytest

COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini-2024-07-18",
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}
CHOICES = {
    "tool_call": {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"city": "Macau"}',
                    },
                }
            ],
        },
    },
    "text": {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "sunny"},
    },
}


@pytest.fixture
def openai_client():
    """Build an OpenAI client whose every call returns the named choice."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    def build(choice, asynchronous=False):
        body = {**COMPLETION, "choices": [CHOICES[choice]]}
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, json=body)
        )
        if asynchronous:
            return openai.AsyncOpenAI(
                api_key="test",
                http_client=httpx.AsyncClient(transport=transport),
            )
        return openai.OpenAI(
            api_key="test", http_client=httpx.Client(transport=transport)
        )

    return build
