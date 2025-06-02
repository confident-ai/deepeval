from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from deepeval.prompt import Prompt


class AgentAttributes(BaseModel):
    # input
    input: Union[str, Dict, list]
    # output
    output: Union[str, Dict, list]


class LlmAttributes(BaseModel):
    # input
    input: Union[str, List[Dict[str, Any]]]
    # output
    output: str
    prompt: Optional[Prompt] = None

    # Optional variables
    input_token_count: Optional[int] = Field(
        None, serialization_alias="inputTokenCount"
    )
    output_token_count: Optional[int] = Field(
        None, serialization_alias="outputTokenCount"
    )

    model_config = {"arbitrary_types_allowed": True}


class RetrieverAttributes(BaseModel):
    # input
    embedding_input: str = Field(serialization_alias="embeddingInput")
    # output
    retrieval_context: List[str] = Field(serialization_alias="retrievalContext")

    # Optional variables
    top_k: Optional[int] = Field(None, serialization_alias="topK")
    chunk_size: Optional[int] = Field(None, serialization_alias="chunkSize")


# Don't have to call this manually, will be taken as input and output of function
# Can be overridden by user
class ToolAttributes(BaseModel):
    # input
    input_parameters: Optional[Dict[str, Any]] = Field(
        None, serialization_alias="inputParameters"
    )
    # output
    output: Optional[Any] = None


Attributes = Union[
    AgentAttributes, LlmAttributes, RetrieverAttributes, ToolAttributes
]
