from deepeval.prompt import Prompt

prompt = Prompt(alias="My First Promptt")
prompt.pull(version="00.00.01")
prompt_to_llm = prompt.interpolate()
print(prompt_to_llm)

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

evaluate(
    test_cases=[
        LLMTestCase(
            input="",
            actual_output="",
        )
    ],
    metrics=[AnswerRelevancyMetric()],
    # hyperparameters={"model": "...", "prompt template": prompt},
)
