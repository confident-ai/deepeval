from pydantic import BaseModel, Field
from typing import Optional, Dict, List


def to_snake_case(string: str) -> str:
    return "".join(
        ["_" + i.lower() if i.isupper() else i for i in string]
    ).lstrip("_")


class Golden(BaseModel):
    input: str
    actual_output: Optional[str] = Field(
        None, serialization_alias="actualOutput"
    )
    expected_output: Optional[str] = Field(
        None, serialization_alias="expectedOutput"
    )
    context: Optional[List[str]] = Field(None)
    retrieval_context: Optional[List[str]] = Field(
        None, serialization_alias="retrievalContext"
    )
    additional_metadata: Optional[Dict] = Field(
        None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(None)
    source_file: Optional[str] = Field(None, serialization_alias="sourceFile")


class ConversationalGolden(BaseModel):
    additional_metadata: Optional[Dict] = Field(
        None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(None)
    messages: List[Golden] = Field(
        default_factory=lambda: [],
        validation_alias="goldens",
        serialization_alias="goldens",
    )
