import os
import asyncio
import time
from dotenv import load_dotenv
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai
from deepeval.integrations.pydantic_ai import Agent
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import Golden, EvaluationDataset

load_dotenv()

instrument_pydantic_ai(api_key=os.getenv("CONFIDENT_API_KEY"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

Agent.instrument_all()

answer_relavancy_metric = AnswerRelevancyMetric()
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    metric_collection="test_collection_1",
    trace_attributes={
        "name": "test_name",
        "tags": ["test_tag_1", "test_tag_2"],
        "metadata": {"test_metadata_key": "test_metadata_value"},
        "thread_id": "test_thread_id",
        "user_id": "test_user_id",
    },
    metrics=[answer_relavancy_metric]
)

goldens = [Golden(input="What's 7 * 8?"), Golden(input="What's 7 * 6?")]

def main():
    dataset = EvaluationDataset(goldens=goldens)
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(agent.run(golden.input))
        dataset.evaluate(task)


if __name__ == "__main__":
    main()
    time.sleep(7)