from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
from deepeval.test_case import LLMTestCase, ConversationalTestCase, ToolCall


class APIEvaluate(BaseModel):
    metric_collection: str = Field(alias="metricCollection")
    llm_test_cases: Optional[List[LLMTestCase]] = Field(alias="llmTestCases")
    conversational_test_cases: Optional[List[ConversationalTestCase]] = Field(
        alias="conversationalTestCases"
    )


class SendTestCaseResponseApiBody(BaseModel):
    actual_output: Optional[str] = Field(
        default=None, serialization_alias="actualOutput"
    )
    retrieval_context: Optional[List[str]] = Field(
        default=None, serialization_alias="retrievalContext"
    )
    tools_called: Optional[List[ToolCall]] = Field(
        default=None, serialization_alias="toolsCalled"
    )
    expected_tools: Optional[List[ToolCall]] = Field(
        default=None, serialization_alias="expectedTools"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None)
