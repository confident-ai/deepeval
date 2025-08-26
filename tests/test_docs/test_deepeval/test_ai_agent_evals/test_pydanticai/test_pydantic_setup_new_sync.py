import time
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent

instrument_pydantic_ai()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    # metric_collection="test_collection_1",
)

def main():
    result = agent.run_sync(
        'Where does "hello world" come from?',
        metric_collection="test_collection_1",
        name="test_trace_name_1",
        tags=["test_tag_1", "test_tag_2"],
        metadata={"test_key_1": "test_value_1"},
        thread_id="test_thread_id_1",
    )
    print(result)

if __name__ == "__main__":
    main()
    time.sleep(10)
