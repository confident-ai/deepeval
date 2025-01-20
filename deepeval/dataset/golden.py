from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from deepeval.test_case import ToolCall


class Golden(BaseModel):
    input: str
    actual_output: Optional[str] = Field(
        default=None, serialization_alias="actualOutput"
    )
    expected_output: Optional[str] = Field(
        default=None, serialization_alias="expectedOutput"
    )
    context: Optional[List[str]] = Field(default=None)
    retrieval_context: Optional[List[str]] = Field(
        default=None, serialization_alias="retrievalContext"
    )
    additional_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(default=None)
    tools_called: Optional[List[ToolCall]] = Field(
        default=None, serialization_alias="toolsCalled"
    )
    expected_tools: Optional[List[ToolCall]] = Field(
        default=None, serialization_alias="expectedTools"
    )
    source_file: Optional[str] = Field(
        default=None, serialization_alias="sourceFile"
    )


class ConversationalGolden(BaseModel):
    additional_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(default=None)
    turns: List[Golden] = Field(
        default_factory=lambda: [],
        serialization_alias="goldens",
    )
