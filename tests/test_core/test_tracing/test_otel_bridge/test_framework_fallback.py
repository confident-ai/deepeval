"""The published SDK remains usable without the unpublished embedding package."""

from openai import _base_client

httpx = getattr(_base_client, "httpx2", None) or _base_client.httpx

from deepeval.tracing import observe
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import LlmSpan


def test_openai_without_confident_trace(monkeypatch):
    from deepeval.tracing.otel import frameworks

    monkeypatch.setattr(frameworks, "available", lambda: False)
    from deepeval.openai import OpenAI
    from deepeval.openai.patch import (
        patch_openai_classes,
        unpatch_openai_classes,
    )

    patch_openai_classes()
    response = {
        "id": "local",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        },
    }

    @observe()
    def run():
        with OpenAI(
            api_key="local",
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=response)
                )
            ),
        ) as client:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "question"}],
            )

    try:
        assert run().choices[0].message.content == "answer"
        trace = trace_manager.post_trace.call_args.args[0]
        model = trace.root_spans[0].children[0]
        assert isinstance(model, LlmSpan)
        assert not model._otel_bridge
        assert not frameworks.enabled("openai")
    finally:
        unpatch_openai_classes()
