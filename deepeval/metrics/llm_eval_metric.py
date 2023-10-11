import json
from typing import Optional, Callable
from deepeval.metrics.metric import Metric
from deepeval.test_case import LLMTestCase
import openai
from pydantic import BaseModel


def make_chat_completion_request(prompt: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


class LLMEvalMetricResponse(BaseModel):
    score: float
    reason: str


class LLMEvalMetric(Metric):
    def __init__(
        self,
        name: str,
        criteria: str,
        completion_function: Callable = make_chat_completion_request,
        prompt_template: Optional[str] = None,
        minimum_score: float = 0.5,
    ):
        self.criteria = criteria
        self.name = name
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = """For the following criteria, evaluate the text, state the reason and then return a score in a JSON with the key `reason` and `score` out of 100 with 100 being that it follows the criteria and 1 being that it does not.

Criteria: {criteria}
Text: {text}

Respond in JSON format in 1 single line without white spaces.
JSON:"""
        self.minimum_score = minimum_score
        self.completion_function = completion_function

    @property
    def __name__(self):
        return self.name

    def measure(self, test_case: LLMTestCase, include_reason: bool = False):
        """Measure out the LLM evaluated metric."""
        # Measure the test case
        prompt: dict = self.prompt_template.format(
            text=test_case.output, criteria=self.criteria
        )
        output: str = self.completion_function(prompt)
        response = json.loads(output.strip())
        score = float(response["score"]) / 100
        reason = response["reason"]
        self.success = score >= self.minimum_score
        if include_reason:
            return {"score": score, "reason": reason}
        return score

    def is_successful(self):
        return self.success
