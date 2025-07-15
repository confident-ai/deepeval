from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.evaluate import dataset, test_run
from deepeval.dataset import Golden
from deepeval.openai import AsyncOpenAI
import asyncio

from tests.integrations.openai.resources import (
    async_llm_app,
    CHAT_TOOLS,
    RESPONSE_TOOLS,
)

goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
] * 20


def test_end_to_end_evaluation():
    openai_client = AsyncOpenAI()
    for golden in dataset(goldens=goldens):
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
            )
        )
        test_run.append(task)

    # for golden in dataset(goldens=goldens):
    #     task = asyncio.create_task(
    #         openai_client.responses.create(
    #             model="gpt-4o",
    #             instructions="You are a helpful chatbot. Always generate a string response.",
    #             input=golden.input,
    #             tools=RESPONSE_TOOLS,
    #             metrics=[AnswerRelevancyMetric(), BiasMetric()],
    #         )
    #     )
    #     test_run.append(task)


def test_component_level_loop():
    for golden in dataset(goldens=goldens):
        task = asyncio.create_task(
            async_llm_app(golden.input, completion_mode="chat")
        )
        test_run.append(task)

    for golden in dataset(goldens=goldens):
        task = asyncio.create_task(
            async_llm_app(golden.input, completion_mode="response")
        )
        test_run.append(task)


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
    # test_component_level_loop()
    # asyncio.run(test_tracing())
