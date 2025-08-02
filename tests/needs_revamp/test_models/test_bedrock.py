from deepeval.models.llms.amazon_bedrock_model import AmazonBedrockModel
from deepeval.metrics import BiasMetric, FaithfulnessMetric
from deepeval.synthesizer import Synthesizer
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import AsyncConfig
from deepeval import evaluate

from pydantic import BaseModel
import asyncio

model = AmazonBedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-east-1",
    aws_access_key_id="...",
    aws_secret_access_key="...",
    temperature=0.5,
    input_token_cost=0.0,
    output_token_cost=0.0,
)


class ModelResponseSchema(BaseModel):
    summary: str
    timestamp: str
    confidence_score: float


simple_prompt = "How are you today?"
structured_prompt = """
Please respond in the following JSON format:
{
  "summary": "A brief description of your current state or mood.",
  "timestamp": "The current time in ISO 8601 format.",
  "confidence_score": "A confidence score between 0 and 1, indicating how certain you are of your response."
}

Question: "How are you today?"
"""


async def test_async():
    tasks = [
        model.a_generate(simple_prompt, schema=None),
        model.a_generate(structured_prompt, schema=ModelResponseSchema),
        model.a_generate(simple_prompt, schema=None),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("=== Async a_generate results ===")
    for idx, result in enumerate(results, start=1):
        if isinstance(result, Exception):
            print(f"Task {idx} failed: {result}")
        else:
            response, cost = result
            print(f"Task {idx} → response: {response!r}, cost: {cost:.4f}")


def test_sync():
    print("\n=== Sync generate results ===")
    tests = [
        (simple_prompt, None),
        (structured_prompt, ModelResponseSchema),
    ]
    for idx, (prompt, schema) in enumerate(tests, start=1):
        try:
            response, cost = model.generate(prompt, schema=schema)
            print(f"Sync {idx} → response: {response!r}, cost: {cost:.4f}")
        except Exception as e:
            print(f"Sync {idx} failed: {e}")


def test_evaluate():
    test_case = LLMTestCase(
        input="What is this again?",
        actual_output="this is a latte",
        expected_output="this is a mocha",
        retrieval_context=["I love coffee"],
        context=["I love coffee"],
    )
    evaluate(
        test_cases=[test_case, test_case, test_case],
        metrics=[FaithfulnessMetric(model=model), BiasMetric(model=model)],
    )


if __name__ == "__main__":
    # test_sync()
    test_evaluate()
    # test_case = LLMTestCase(
    #     input="What is this again?",
    #     actual_output="this is a latte",
    #     expected_output="this is a mocha",
    #     retrieval_context=["I love coffee"],
    #     context=["I love coffee"]
    # )
    # FaithfulnessMetric(model=model, async_mode=False).measure(test_case)
