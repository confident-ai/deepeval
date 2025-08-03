from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.openai import AsyncOpenAI
from deepeval.dataset import EvaluationDataset, Golden
import asyncio

from tests.integrations.openai.resources import (
    async_llm_app,
    CHAT_TOOLS,
)

goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
] * 10


def test_end_to_end_evaluation():
    openai_client = AsyncOpenAI()
    dataset = EvaluationDataset(goldens=goldens)
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(
            openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful chatbot. Always generate a string response.",
                    },
                    {"role": "user", "content": golden.input},
                ],
                tools=CHAT_TOOLS,
                metrics=[AnswerRelevancyMetric(), BiasMetric()],
            ),
        )
        dataset.evaluate(task)


def test_component_level_loop():
    dataset = EvaluationDataset(goldens=goldens)
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(
            async_llm_app(golden.input, completion_mode="chat")
        )
        dataset.evaluate(task)

    for golden in dataset.evals_iterator():
        task = asyncio.create_task(
            async_llm_app(golden.input, completion_mode="response")
        )
        dataset.evaluate(task)


async def test_tracing():
    await async_llm_app(
        "What is the weather in Bogotá, Colombia?", completion_mode="chat"
    )
    await async_llm_app(
        "What is the weather in Paris, France?", completion_mode="chat"
    )
    await async_llm_app(
        "What is the weather in Bogotá, Colombia?", completion_mode="response"
    )
    await async_llm_app(
        "What is the weather in Paris, France?", completion_mode="response"
    )


##############################################
# Test Everything
##############################################

if __name__ == "__main__":
    test_end_to_end_evaluation()
    test_component_level_loop()
    asyncio.run(test_tracing())
