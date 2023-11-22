from typing import Optional, List
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.templates import (
    evaluation_steps_template,
    evaluation_results_template,
)
from deepeval.chat_completion.retry import call_openai_with_retry
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI


class LLMEvalMetricResponse(BaseModel):
    score: float


class LLMEvalMetric(BaseMetric):
    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_params: List[LLMTestCaseParams],
        evaluation_steps: str = "",
        model: Optional[str] = "gpt-4-1106-preview",
        minimum_score: float = 0.5,
        **kwargs,
    ):
        self.criteria = criteria
        self.name = name
        self.model = model
        self.evaluation_steps = evaluation_steps
        self.evaluation_params = evaluation_params
        self.minimum_score = minimum_score
        self.deployment_id = None
        if "deployment_id" in kwargs:
            self.deployment_id = kwargs["deployment_id"]

    def measure(self, test_case: LLMTestCase):
        """LLM evaluated metric based on the GEval framework: https://arxiv.org/pdf/2303.16634.pdf"""

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
        self.score = float(score) * 2 / 10
        self.success = score >= self.minimum_score
        return self.score

    def is_successful(self):
        return self.success

    def generate_evaluation_steps(self):
        prompt: dict = evaluation_steps_template.format(criteria=self.criteria)

        model_kwargs = {}
        if self.deployment_id is not None:
            model_kwargs["deployment_id"] = self.deployment_id

        chat_completion = ChatOpenAI(
            model_name=self.model, model_kwargs=model_kwargs
        )

        res = call_openai_with_retry(lambda: chat_completion.invoke(prompt))
        return res.content

    def evaluate(self, test_case: LLMTestCase):
        text = """"""

        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            text += f"{param.value}: {value} \n\n"

        prompt: dict = evaluation_results_template.format(
            evaluation_steps=self.evaluation_steps,
            text=text,
        )

        model_kwargs = {
            "top_p": 1,
            "frequency_penalty": 0,
            "stop": None,
            "presence_penalty": 0,
        }
        if self.deployment_id is not None:
            model_kwargs["deployment_id"] = self.deployment_id

        chat_completion = ChatOpenAI(
            model_name=self.model, max_tokens=5, n=20, model_kwargs=model_kwargs
        )

        res = call_openai_with_retry(
            lambda: chat_completion.generate_prompt(
                [chat_completion._convert_input(prompt)]
            )
        )

        total_scores = 0
        count = 0

        for content in res.generations[0]:
            try:
                total_scores += float(content.message.content)
                count += 1
            except:
                pass

        return total_scores / count

    @property
    def __name__(self):
        return self.name
