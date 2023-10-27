from typing import Optional, List
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.templates import (
    evaluation_steps_template,
    evaluation_results_template,
)
from deepeval.types import LLMTestCaseParams
from deepeval.chat_completion.retry import call_openai_with_retry
from pydantic import BaseModel
import openai


class LLMEvalMetricResponse(BaseModel):
    score: float


class LLMEvalMetric(BaseMetric):
    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_params: List[LLMTestCaseParams],
        model: Optional[str] = "gpt-4",
        minimum_score: float = 0.5,
    ):
        self.criteria = criteria
        self.name = name
        self.model = model
        self.evaluation_steps = ""
        self.evaluation_params = evaluation_params

        self.minimum_score = minimum_score

    @property
    def __name__(self):
        return self.name

    def measure(self, test_case: LLMTestCase):
        """Measure out the LLM evaluated metric."""
        # Measure the test case
        for param in self.evaluation_params:
            if (
                not hasattr(test_case, param.value)
                or getattr(test_case, param.value) is None
            ):
                raise ValueError(
                    f"Test case is missing the required attribute: {param.value}"
                )

        if self.evaluation_steps == "":
            self.evaluation_steps = self.generate_evaluation_steps()

        score = self.evaluate(test_case)
        score = float(score) * 2 / 10

        self.success = score >= self.minimum_score
        return score

    def is_successful(self):
        return self.success

    def generate_evaluation_steps(self):
        prompt: dict = evaluation_steps_template.format(criteria=self.criteria)

        res = call_openai_with_retry(
            lambda: openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        )

        return res.choices[0].message.content

    def evaluate(self, test_case: LLMTestCase):
        text = """"""

        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            text += f"{param.value}: {value} \n\n"

        prompt: dict = evaluation_results_template.format(
            evaluation_steps=self.evaluation_steps,
            text=text,
        )

        res = call_openai_with_retry(
            lambda: openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                # logprobs=5,
                n=20,
            )
        )

        total_scores = 0
        count = 0

        for content in res.choices:
            try:
                total_scores += float(content.message.content)
                count += 1
            except:
                pass

        return total_scores / count
