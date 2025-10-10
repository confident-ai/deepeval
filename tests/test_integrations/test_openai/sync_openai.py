from deepeval.openai import OpenAI
from deepeval.tracing import trace
from deepeval.prompt import Prompt

client = OpenAI()

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")

def test_sync_openai_without_trace():
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )

def test_sync_openai_with_trace():

    with trace(
        prompt=prompt,
        thread_id="test_thread_id_1",
        llm_metric_collection="test_collection_1",
        name="test_name_1",
        tags=["test_tag_1"],
        metadata={"test_metadata_1": "test_value_1"},
        user_id="test_user_id_1",
    ):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

def test_sync_response_create_without_trace():
    response = client.responses.create(
        model="gpt-4o",
        instructions="You are a helpful assistant. Always generate a string response.",
        input="Hello, how are you?",
    )

def test_sync_response_create_with_trace():
    with trace(
        prompt=prompt,
        thread_id="test_thread_id_1",
        llm_metric_collection="test_collection_1",
        name="test_name_1",
        tags=["test_tag_1"],
        metadata={"test_metadata_1": "test_value_1"},
        user_id="test_user_id_1",
    ):
        response = client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant. Always generate a string response.",
            input="Hello, how are you?",
        )

if __name__ == "__main__":
    test_sync_openai_without_trace()
    test_sync_openai_with_trace()
    test_sync_response_create_without_trace()
    test_sync_response_create_with_trace()