import time

from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent

instrument_pydantic_ai()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
)

result = agent.run_sync(
    "What are the LLMs?",
    trace_name="test_trace_name_1",
    trace_tags=["test_tag_1", "test_tag_2"],
    trace_metadata={"test_key_1": "test_value_1"},
    trace_thread_id="test_thread_id_1",
    trace_user_id="test_user_id_1",
)

print(result)
time.sleep(10)  # wait for the trace to be posted
