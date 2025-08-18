from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent

from deepeval.metrics import AnswerRelevancyMetric

instrument_pydantic_ai()

# Agent.instrument_all()

answer_relavancy_metric = AnswerRelevancyMetric()
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    # metrics=[answer_relavancy_metric],
    trace_attributes={
        "name": "test_trace",
        "tags": ["tag1", "tag2"],
        "metadata": {"key": "value"},
        "thread_id": "123",
        "user_id": "456",
    },
)

# run for testing (not needed for docs)
result = agent.run_sync("What are the LLMs?")
print(result)
