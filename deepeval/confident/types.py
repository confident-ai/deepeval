from pydantic import BaseModel, Field
from typing import List, Union

from deepeval.test_case import LLMTestCase, ConversationalTestCase


class ConfidentEvaluateRequestData(BaseModel):
    experiment_name: str = Field(alias="experimentName")
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]] = Field(
        alias="testCases"
    )


class ConfidentEvaluateResponseData(BaseModel):
    link: str
