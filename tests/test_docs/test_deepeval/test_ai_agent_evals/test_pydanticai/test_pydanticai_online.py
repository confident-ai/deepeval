import time
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent


from deepeval.metrics import AnswerRelevancyMetric

instrument_pydantic_ai()

Agent.instrument_all()

answer_relavancy_metric = AnswerRelevancyMetric()
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    metric_collection="test_collection_1",
)

# run for testing (not needed for docs)
result = agent.run_sync('Where does "hello world" come from?')

time.sleep(10)