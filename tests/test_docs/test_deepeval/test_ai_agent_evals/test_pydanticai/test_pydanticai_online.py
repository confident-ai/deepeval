import time

from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent

instrument_pydantic_ai()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
)

result = agent.run_sync(
    "What are the LLMs?",
    metric_collection="test_collection_1",
)

print(result)
time.sleep(10)  # wait for the trace to be posted
