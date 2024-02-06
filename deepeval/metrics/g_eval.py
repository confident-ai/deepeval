import json
from typing import Optional, List, Tuple, Union
from langchain_core.language_models import BaseChatModel

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.templates import (
    evaluation_steps_template,
    evaluation_results_template,
)
from deepeval.utils import trimToJson
from deepeval.models import GPTModel, DeepEvalBaseModel

from pydantic import BaseModel


class GEvalResponse(BaseModel):
    score: float
    reason: str


class GEval(BaseMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: List[LLMTestCaseParams],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        model: Optional[Union[str, DeepEvalBaseModel, BaseChatModel]] = None,
        threshold: float = 0.5,
    ):
        self.name = name
        self.evaluation_params = evaluation_params

        # Check if both criteria and evaluation_steps are not None at the same time
        if criteria is None and evaluation_steps is None:
            raise ValueError(
                "Either 'criteria' or 'evaluation_steps' must be provided, but not both None."
            )

        # Check if criteria is provided, it cannot be an empty string
        if criteria is not None and not criteria.strip():
            raise ValueError("Criteria provided cannot be an empty string.")

        # Check if evaluation_steps is provided, it cannot be an empty list
        if evaluation_steps is not None and len(evaluation_steps) == 0:
            raise ValueError(
                "Evaluation steps must not be an empty list. Either omit evaluation steps or include a non-empty list of steps."
            )

        self.criteria = criteria
        if isinstance(model, DeepEvalBaseModel):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.evaluation_steps = evaluation_steps
        self.threshold = threshold

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

        if self.evaluation_steps is None:
            json_output = trimToJson(self.generate_evaluation_steps())
            data = json.loads(json_output)
            self.evaluation_steps = data["steps"]

        score, reason = self.evaluate(test_case)
        self.reason = reason
        self.score = float(score) / 10
        self.success = score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    def generate_evaluation_steps(self):
        prompt: dict = evaluation_steps_template.format(criteria=self.criteria)

        res = self.model(prompt)

        return res

    def evaluate(self, test_case: LLMTestCase) -> Tuple[int, str]:
        text = """"""

        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            text += f"{param.value}: {value} \n\n"

        prompt: dict = evaluation_results_template.format(
            evaluation_steps=self.numbered_evaluation_steps(),
            text=text,
        )

        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)

        return data["score"], data["reason"]

    def numbered_evaluation_steps(self):
        evaluation_steps = """"""
        for index, string in enumerate(self.evaluation_steps, start=1):
            evaluation_steps += f"{index}. {string}\n"

        return evaluation_steps

    @property
    def __name__(self):
        return self.name
