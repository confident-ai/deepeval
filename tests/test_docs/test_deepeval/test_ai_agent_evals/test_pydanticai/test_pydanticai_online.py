import time
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent

instrument_pydantic_ai()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    metric_collection="test_collection_1"
)

result = agent.run_sync('Where does "hello world" come from?')

time.sleep(10)
