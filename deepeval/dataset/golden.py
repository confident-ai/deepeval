from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class Golden(BaseModel):
    input: str
    actual_output: Optional[str] = Field(None, alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    context: Optional[list] = Field(None)
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    additional_metadata: Optional[Dict] = Field(
        None, alias="additionalMetadata"
    )
    source_file: Optional[str] = Field(None, alias="sourceFile")


class ConversationalGolden(BaseModel):
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    additional_metadata: Optional[Dict] = Field(
        None, alias="additionalMetadata"
    )
    goldens: List[Golden] = Field(default_factory=lambda: [])
