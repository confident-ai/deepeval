from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from typing import List
from pydantic import BaseModel
from deepeval.api import Api


class JudgementalGPTResponse(BaseModel):
    score: float
    reason: str


class JudgementalGPTRequest(BaseModel):
    text: str
    criteria: str


class JudgementalGPT(BaseMetric):
    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_params: List[LLMTestCaseParams],
        minimum_score: float = 0.5,
    ):
        self.criteria = criteria
        self.name = name
        self.evaluation_params = evaluation_params
        self.minimum_score = minimum_score
        self.success = None
        self.reason = None

    @property
    def __name__(self):
        return self.name

    def measure(self, test_case: LLMTestCase):
        text = """"""
        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            text += f"{param.value}: {value} \n\n"

        judgemental_gpt_request_data = JudgementalGPTRequest(
            text=text, criteria=self.criteria
        )

        try:
            body = judgemental_gpt_request_data.model_dump(
                by_alias=True, exclude_none=True
            )
        except AttributeError:
            body = judgemental_gpt_request_data.dict(
                by_alias=True, exclude_none=True
            )
        api = Api()
        result = api.post_request(
            endpoint="/v1/judgemental-gpt",
            body=body,
        )
        response = JudgementalGPTResponse(
            score=result["score"],
            reason=result["reason"],
        )
        self.reason = response.reason
        self.score = response.score / 10
        self.success = self.score >= self.minimum_score

        return self.score

    def is_successful(self):
        return self.success
