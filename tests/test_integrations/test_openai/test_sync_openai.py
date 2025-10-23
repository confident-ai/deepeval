from openai import OpenAI
from deepeval.tracing import trace, LlmSpanContext
from deepeval.prompt import Prompt
from tests.test_integrations.utils import assert_trace_json, generate_trace_json
import os
import pytest


client = OpenAI()

prompt = Prompt(alias="asd")
prompt._version = "00.00.01"

_current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skip
def test_sync_openai_without_trace():
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )


@assert_trace_json(
    json_path=os.path.join(_current_dir, "test_sync_openai_with_trace.json")
)
def test_sync_openai_with_trace():

    with trace(
        llm_span_context=LlmSpanContext(
            prompt=prompt,
            metric_collection="test_collection_1",
        ),
        thread_id="test_thread_id_1",
        name="test_name_1",
        tags=["test_tag_1"],
        metadata={"test_metadata_1": "test_value_1"},
        user_id="test_user_id_1",
    ):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )


@assert_trace_json(
    json_path=os.path.join(
        _current_dir, "test_sync_response_create_without_trace.json"
    )
)
def test_sync_response_create_without_trace():
    client.responses.create(
        model="gpt-4o",
        instructions="You are a helpful assistant. Always generate a string response.",
        input="Hello, how are you?",
    )


@pytest.mark.skip
def test_sync_response_create_with_trace():
    with trace(
        llm_span_context=LlmSpanContext(
            prompt=prompt,
            metric_collection="test_collection_1",
        ),
        thread_id="test_thread_id_1",
        name="test_name_1",
        tags=["test_tag_1"],
        metadata={"test_metadata_1": "test_value_1"},
        user_id="test_user_id_1",
    ):
        client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant. Always generate a string response.",
            input="Hello, how are you?",
        )


def generate_all_json_dumps():
    # test_sync_openai_without_trace()
    test_sync_openai_with_trace()
    test_sync_response_create_without_trace()
    # test_sync_response_create_with_trace()
