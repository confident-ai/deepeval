from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Optional, Dict, List
from deepeval.test_case import ToolCall, Turn, MLLMImage


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
    name: Optional[str] = Field(default=None)
    custom_column_key_values: Optional[Dict[str, str]] = Field(
        default=None, serialization_alias="customColumnKeyValues"
    )
    multimodal: bool = Field(False, exclude=True)
    _dataset_rank: Optional[int] = PrivateAttr(default=None)
    _dataset_alias: Optional[str] = PrivateAttr(default=None)
    _dataset_id: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_is_multimodal(self):
        import re

        if self.multimodal is True:
            return self

        pattern = r"\[DEEPEVAL:IMAGE:(.*?)\]"
        self.multimodal = (
            any(
                [
                    (
                        re.search(pattern, self.input) is not None
                        if self.input
                        else False
                    ),
                    (
                        re.search(pattern, self.actual_output) is not None
                        if self.actual_output
                        else False
                    ),
                ]
            )
            if isinstance(self.input, str)
            else self.multimodal
        )

        return self


class ConversationalGolden(BaseModel):
    scenario: str
    expected_outcome: Optional[str] = Field(
        None, serialization_alias="expectedOutcome"
    )
    user_description: Optional[str] = Field(
        None, serialization_alias="userDescription"
    )
    context: Optional[List[str]] = Field(default=None)
    additional_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    custom_column_key_values: Optional[Dict[str, str]] = Field(
        default=None, serialization_alias="customColumnKeyValues"
    )
    turns: Optional[List[Turn]] = Field(default=None)
    multimodal: bool = Field(False, exclude=True)
    _dataset_rank: Optional[int] = PrivateAttr(default=None)
    _dataset_alias: Optional[str] = PrivateAttr(default=None)
    _dataset_id: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_is_multimodal(self):
        import re

        if self.multimodal is True:
            return self

        pattern = r"\[DEEPEVAL:IMAGE:(.*?)\]"
        self.multimodal = (
            any(
                [
                    re.search(pattern, turn.content) is not None
                    for turn in self.turns
                ]
            )
            if self.turns
            else self.multimodal
        )

        return self
