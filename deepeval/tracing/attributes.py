from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from deepeval.prompt import Prompt


class AgentAttributes(BaseModel):
    # input
    input: Union[str, Dict, list]
    # output
    output: Union[str, Dict, list]


class LlmToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    id: Optional[str] = None


class LlmOutput(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[LlmToolCall]] = None


class LlmAttributes(BaseModel):
    # input
    input: Union[str, List[Dict[str, Any]]]
    # output
    output: Union[str, LlmOutput, List[Dict[str, Any]]]
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


class GraphAttributes(BaseModel):
    """Attributes specific to LangGraph execution"""
    graph_id: str
    compilation_time: Optional[float] = None
    total_nodes: int
    execution_strategy: str
    state_schema: Optional[Dict[str, Any]] = None


class NodeAttributes(BaseModel):
    """Attributes specific to LangGraph node execution"""
    node_id: str
    node_type: str
    dependencies: List[str]
    execution_order: int
    conditional_logic: Optional[str] = None
    parallel_group: Optional[str] = None


class StateTransitionAttributes(BaseModel):
    """Attributes specific to LangGraph state transitions"""
    from_node: str
    to_node: str
    state_changes: Dict[str, Any]
    transition_condition: Optional[str] = None
    routing_decision: Optional[str] = None


Attributes = Union[
    AgentAttributes, LlmAttributes, RetrieverAttributes, ToolAttributes,
    GraphAttributes, NodeAttributes, StateTransitionAttributes
]
