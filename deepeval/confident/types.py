from typing import List, Union

from pydantic import BaseModel, Field

from deepeval.test_case import ConversationalTestCase, LLMTestCase


class ConfidentEvaluateRequestData(BaseModel):
    metric_collection: str = Field(alias="metricCollection")
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]] = Field(
        alias="testCases"
    )


class ConfidentEvaluateResponseData(BaseModel):
    link: str
