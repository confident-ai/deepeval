from pydantic import BaseModel, Field
from typing import Optional, List
from deepeval.test_case import LLMTestCase, ConversationalTestCase


class APIEvaluate(BaseModel):
    metric_collection: str = Field(alias="metricCollection")
    llm_test_cases: Optional[List[LLMTestCase]] = Field(alias="llmTestCases")
    conversational_test_cases: Optional[List[ConversationalTestCase]] = Field(
        alias="conversationalTestCases"
    )
