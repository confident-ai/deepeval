from typing import List
from openai import OpenAI

from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

client = OpenAI()


def your_llm_app(input: str):
    def retriever(input: str):
        return ["Hardcoded text chunks from your vector database"]

    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str, retrieved_chunks: List[str]):
        res = (
            client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Use the provided context to answer the question.",
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(retrieved_chunks)
                        + "\n\nQuestion: "
                        + input,
                    },
                ],
            )
            .choices[0]
            .message.content
        )

        # Create test case at runtime
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=res)
        )

        return res

    return generator(input, retriever(input))


print(your_llm_app("How are you?"))


#################################

# from somewhere import your_async_llm_app # Replace with your async LLM app
# from deepeval.dataset import EvaluationDataset, Golden
# import asyncio

# dataset = EvaluationDataset(goldens=[Golden(input="...")])

# for golden in dataset.evals_iterator():
#     # Create task to invoke your async LLM app
#     task = asyncio.create_task(your_async_llm_app(golden.input))
#     dataset.evaluate(task)

##################################

# from somewhere import your_llm_app # Replace with your LLM app
# import pytest
# from deepeval.dataset import Golden
# from deepeval import assert_test

# # Goldens from your dataset
# goldens = [Golden(input="...")]

# # Loop through goldens using pytest
# @pytest.mark.parametrize("golden", goldens)
# def test_llm_app(golden: Golden):
#     assert_test(golden=golden, observed_callback=your_llm_app)
