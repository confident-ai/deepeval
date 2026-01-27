import os
import pytest

from anthropic import Anthropic
from deepeval.prompt import Prompt
from deepeval.tracing import LlmSpanContext, trace
from tests.test_integrations.utils import assert_trace_json


client = Anthropic()

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"

_current_dir = os.path.dirname(os.path.abspath(__file__))


@assert_trace_json(
    json_path=os.path.join(
        _current_dir, "test_sync_messages_create_without_trace.json"
    )
)
def test_sync_messages_create_without_trace():
    client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system="You are a helpful assistant. Always generate a string response.",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )


@pytest.mark.skip
def test_sync_messages_create_with_trace():
    with trace(
        llm_span_context=LlmSpanContext(
            prompt=prompt,
            metric_collection="test_collection_1",
        ),
        name="test_name_1",
        tags=["test_tag_1"],
        metadata={"test_metadata_1": "test_value_1"},
        user_id="test_user_id_1",
        thread_id="test_thread_id_1",
    ):
        client.responses.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system="You are a helpful assistant. Always generate a string response.",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )


def generate_all_json_dumps():
    test_sync_messages_create_with_trace()
    test_sync_messages_create_without_trace()
