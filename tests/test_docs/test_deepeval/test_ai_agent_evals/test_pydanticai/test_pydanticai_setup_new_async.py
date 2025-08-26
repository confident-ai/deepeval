import asyncio
import time
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent
from deepeval.metrics import AnswerRelevancyMetric

instrument_pydantic_ai()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    # metric_collection="test_collection_1",
)

answer_relavancy_metric = AnswerRelevancyMetric()

async def main():
    result = await agent.run(
        'Where does "hello world" come from?',
        metric_collection="test_collection_1",
        name="test_trace_name_1",
        tags=["test_tag_1", "test_tag_2"],
        metadata={"test_key_1": "test_value_1"},
        thread_id="test_thread_id_1",
    )
    print(result)

from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's 7 * 8?"),
        Golden(input="What's 7 * 6?"),
    ]
)


if __name__ == "__main__":
    # for golden in dataset.evals_iterator():
    #     task = asyncio.create_task(agent.run(
    #         golden.input,
    #         metrics=[answer_relavancy_metric],
    #     ))
    #     dataset.evaluate(task)
    
    asyncio.run(main())
    time.sleep(10)
