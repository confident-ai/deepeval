from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Union, Optional, Dict
from time import perf_counter
from pydantic import BaseModel, Field
import inspect

from deepeval.utils import dataclass_to_dict
from deepeval.event import track

########################################################
### Trace Types ########################################
########################################################

class TraceProvider(Enum):
    LLAMA_INDEX = "LLAMA_INDEX"
    DEFAULT = "DEFAULT"
    CUSTOM = "CUSTOM"
    HYBRID = "HYBRID"

class TraceType(Enum):
    AGENT = "Agent"
    CHAIN = "Chain"
    EMBEDDING = "Embedding"
    LLM = "LLM"
    QUERY = "Query"
    RERANKING = "Reranking"
    RETRIEVER = "Retriever"
    SYNTHESIZE = "Synthesize"
    TOOL = "Tool"

class LlamaIndexTraceType(Enum):
    AGENT = "Agent"
    CHAIN = "Chain"
    EMBEDDING = "Embedding"
    LLM = "LLM"
    QUERY = "Query"
    RERANKING = "Reranking"
    RETRIEVER = "Retriever"
    SYNTHESIZE = "Synthesize"
    TOOL = "Tool"

class TraceStatus(Enum):
    SUCCESS = "Success"
    ERROR = "Error"

class RetrievalNode(BaseModel):
    content: str
    # Optional variables
    id: Optional[str] = None
    score: Optional[float] = None
    source_file: Optional[str] = None

########################################################
### Metadata Types #####################################
########################################################

class AgentMetadata(BaseModel):
    input: str
    output: str
    name: str
    description: str

class ChainMetadata(BaseModel):
    input: str
    output: str
    # Optional variables
    prompt_template: Optional[str] = Field(None, serialization_alias="promptTemplate")

class EmbeddingMetadata(BaseModel):
    embedding_text: str = Field(None, serialization_alias="embeddingText")
    # Optional variables
    model: Optional[str] = None
    embedding_length: Optional[int] = Field(None, serialization_alias="embeddingLength")

class LlmMetadata(BaseModel):
    input_str: str = Field(None, serialization_alias="inputStr")
    output_str: str = Field(None, serialization_alias="inputStr")
    # Optional variables
    model: Optional[str] = None
    total_token_count: Optional[int] = Field(None, serialization_alias="totalTokenCount")
    prompt_token_count: Optional[int] = Field(None, serialization_alias="promptTokenCount")
    completion_token_count: Optional[int] = Field(None, serialization_alias="completionTokenCount")
    prompt_template: Optional[str] = Field(None, serialization_alias="promptTemplate")
    prompt_template_variables: Optional[Dict[str, str]] = Field(
        None, serialization_alias="promptTemplateVariables"
    )

class QueryMetadata(BaseModel):
    input: str
    output: str

class RetrieverMetadata(BaseModel):
    query_str: str
    nodes: List[RetrievalNode]
    # Optional variables
    top_k: Optional[int] = Field(None, serialization_alias="topK")
    average_chunk_size: Optional[int] = Field(None, serialization_alias="averageChunkSize")
    top_score: Optional[float] = Field(None, serialization_alias="topScore")
    similarity_scorer: Optional[str] = Field(None, serialization_alias="similarityScorer")

class RerankingMetadata(BaseModel):
    input_nodes: List[RetrievalNode]
    output_nodes: List[RetrievalNode]
    # Optional variables
    model: Optional[str] = None
    top_n: Optional[int] = Field(None, serialization_alias="topN")
    batch_size: Optional[int] = Field(None, serialization_alias="batchSize")
    query_str: Optional[str] = Field(None, serialization_alias="queryStr")

class SynthesizeMetadata(BaseModel):
    user_query: str
    response: str
    # Optional variables
    retrieved_context: Optional[str]

class ToolMetadata(BaseModel):
    name: str
    description: str

class GenericMetadata(BaseModel):
    input: Optional[str]
    output: Optional[str]

########################################################
### Trace Types #######################################
########################################################

@dataclass
class BaseTrace:
    type: Union[TraceType, str]
    executionTime: float
    name: str
    status: TraceStatus
    traceProvider: TraceProvider
    traces: List["TraceData"]

@dataclass
class AgentTrace(BaseTrace):
    agentMetadata: AgentMetadata
    type: TraceType

@dataclass
class ChainTrace(BaseTrace):
    chainMetadata: ChainMetadata
    type: TraceType

@dataclass
class EmbeddingTrace(BaseTrace):
    embeddingMetadata: EmbeddingMetadata
    type: TraceType

@dataclass
class LlmTrace(BaseTrace):
    llmMetadata: LlmMetadata
    type: TraceType

@dataclass
class QueryTrace(BaseTrace):
    queryMetadata: QueryMetadata
    type: TraceType
    
@dataclass
class RetrieverTrace(BaseTrace):
    retrieverMetadata: RetrieverMetadata
    type: TraceType

@dataclass
class RerankingTrace(BaseTrace):
    rerankingMetadata: RerankingMetadata
    type: TraceType

@dataclass
class SynthesizeTrace(BaseTrace):
    synthesizeMetadata: SynthesizeMetadata
    type: TraceType

@dataclass
class ToolTrace(BaseTrace):
    toolMetadata: ToolMetadata
    type: TraceType
    
@dataclass
class GenericTrace(BaseTrace):
    genericMetadata: Optional[GenericMetadata] = None
    type: str

Metadata = Union[
    AgentMetadata, ChainMetadata, EmbeddingMetadata, 
    LlmMetadata, QueryMetadata, RerankingMetadata,
    RerankingMetadata, RetrieverMetadata, SynthesizeMetadata,
    ToolMetadata, GenericMetadata
]
TraceData = Union[
    AgentTrace, ChainTrace, EmbeddingTrace, LlmTrace,
    QueryTrace, RerankingTrace, RetrieverTrace,
    SynthesizeTrace, ToolTrace, GenericTrace
]
TraceStack = List[TraceData]

# Context variable to maintain an isolated stack for each async task
trace_stack_var: ContextVar[TraceStack] = ContextVar("trace_stack", default=[])
dict_trace_stack_var = ContextVar("dict_trace_stack", default=None)

########################################################
### ContextVar Managers ################################
########################################################

class TraceManager:
    def get_trace_stack(self):
        return trace_stack_var.get()

    def get_trace_stack_copy(self):
        return trace_stack_var.get().copy()

    def set_trace_stack(self, new_stack):
        trace_stack_var.set(new_stack)

    def clear_trace_stack(self):
        self.set_trace_stack([])

    def pop_trace_stack(self):
        current_stack = self.get_trace_stack_copy()
        if current_stack:
            self.set_trace_stack(current_stack[:-1])

    def append_to_trace_stack(self, trace_instance):
        current_stack = self.get_trace_stack_copy()
        current_stack.append(trace_instance)
        self.set_trace_stack(current_stack)

    def set_dict_trace_stack(self, dict_trace_stack):
        dict_trace_stack_var.set(dict_trace_stack)

    def get_and_reset_dict_trace_stack(self):
        dict_trace_stack = dict_trace_stack_var.get()
        dict_trace_stack_var.set(None)
        return dict_trace_stack

trace_manager = TraceManager()

########################################################
### Tracer #############################################
########################################################

class Tracer:
    # IMPORTANT: framework callback integrations does NOT use this Tracer
    def __init__(self, trace_type: Union[TraceType, str]):
        self.trace_type: TraceType|str = trace_type
        if isinstance(self.trace_type, TraceType):
            self.trace_provider = TraceProvider.DEFAULT
        else:
            self.trace_provider = TraceProvider.CUSTOM
        self.name: str
        self.start_time: float
        self.execution_time: float
        self.status: TraceStatus
        self.error: Optional[Dict[str:Any]] = None
        self.metadata: Optional[Metadata] = None
        self.track_params: Optional[Dict] = None
        self.is_tracking: bool = False

    def __enter__(self):
        # start timer
        self.start_time = perf_counter()

        # create name
        caller_frame = inspect.currentframe().f_back
        func_name = caller_frame.f_code.co_name
        self.name = func_name

        # append trace instance to stack
        trace_instance: BaseTrace = self.create_trace_instance(self.trace_type, self.trace_provider)
        trace_manager.append_to_trace_stack(trace_instance)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Check metadata was set
        if self.metadata and self.trace_provider == TraceProvider.DEFAULT:
            raise ValueError(
                f"`set_parameters` was not called before the end of a {self.trace_type} trace type."
            )

        # Stop the span timing and calculate execution time
        self.execution_time = perf_counter() - self.start_time

        # Check if an exception occurred within the `with` block
        if exc_type is not None:
            self.status = TraceStatus.ERROR
            self.error = {
                "status": "Error",
                "exception_type": exc_type.__name__,
                "message": str(exc_val),
            }
        else:
            self.status = TraceStatus.SUCCESS

        self.update_trace_instance()
        current_trace_stack = trace_manager.get_trace_stack_copy()
        trace_instance = current_trace_stack[-1]

        if len(current_trace_stack) > 1:
            parent_trace = current_trace_stack[-2]
            parent_trace.traces.append(trace_instance)
            trace_manager.set_trace_stack(current_trace_stack)

        if len(current_trace_stack) == 1:
            dict_representation = dataclass_to_dict(current_trace_stack[0])
            trace_manager.set_dict_trace_stack(dict_representation)
            trace_manager.clear_trace_stack()

            print(dict_representation)

            if self.is_tracking:
                track(
                    event_name=self.track_params["event_name"]
                    or self.trace_type,
                    model=self.track_params["model"],
                    input=self.track_params["input"],
                    response=self.track_params["response"],
                    retrieval_context=self.track_params["retrieval_context"],
                    completion_time=self.track_params["completion_time"]
                    or self.execution_time,
                    token_usage=self.track_params["token_usage"],
                    token_cost=self.track_params["token_cost"],
                    distinct_id=self.track_params["distinct_id"],
                    conversation_id=self.track_params["conversation_id"],
                    additional_data=self.track_params["additional_data"],
                    hyperparameters=self.track_params["hyperparameters"],
                    fail_silently=self.track_params["fail_silently"],
                    run_async=self.track_params["run_async"],
                    trace_stack=dict_representation,
                )
        else:
            trace_manager.pop_trace_stack()

    def create_trace_instance(
        self,
        trace_type: Union[TraceType, str],
        trace_provider: TraceProvider,
    ):
        if trace_provider == TraceProvider.DEFAULT:
            if trace_type == TraceType.AGENT:
                return AgentTrace(
                    type=trace_type,
                    traceProvider=trace_provider,
                    executionTime=0,
                    name=self.name,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    agentMetadata=None,
                )
            elif trace_type == TraceType.CHAIN:
                return ChainTrace(
                    type=trace_type,
                    traceProvider=trace_provider,
                    executionTime=0,
                    name=self.name,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    chainMetadata=None,
                )
            elif trace_type == TraceType.EMBEDDING:
                return EmbeddingTrace(
                    type=trace_type,
                    traceProvider=trace_provider,
                    executionTime=0,
                    name=self.name,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    embeddingMetadata=None,
                )
            elif trace_type == TraceType.LLM:
                return LlmTrace(
                    type=trace_type,
                    traceProvider=trace_provider,
                    executionTime=0,
                    name=self.name,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    llmMetadata=None,
                )
            elif trace_type == TraceType.QUERY:
                return QueryTrace(
                    type=trace_type,
                    traceProvider=trace_provider,
                    executionTime=0,
                    name=self.name,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    queryMetadata=None,
                )
            elif trace_type == TraceType.RERANKING:
                return RerankingTrace(
                    type=trace_type,
                    traceProvider=trace_provider,
                    executionTime=0,
                    name=self.name,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    rerankingMetadata=None,
                )
            elif trace_type == TraceType.RETRIEVER:
                return RetrieverTrace(
                    type=trace_type,
                    traceProvider=trace_provider,
                    executionTime=0,
                    name=self.name,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    retrieverMetadata=None,
                )
            elif trace_type == TraceType.SYNTHESIZE:
                return SynthesizeTrace(
                    type=trace_type,
                    traceProvider=trace_provider,
                    executionTime=0,
                    name=self.name,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    synthesizeMetadata=None
                )
            
        elif trace_provider == TraceProvider.CUSTOM:
            return GenericTrace(
                type=trace_type,
                traceProvider=trace_provider,
                executionTime=0,
                name=self.name,
                status=TraceStatus.SUCCESS,
                traces=[],
            )

    def update_trace_instance(self):
        # Get the latest trace instance from the stack
        current_stack = trace_manager.get_trace_stack_copy()
        current_trace = current_stack[-1]

        # update current_trace
        current_trace.executionTime = self.execution_time
        current_trace.status = self.status

        # Assert that the metadata is of the correct type and assign it to the current trace
        trace_mapping = {
            TraceType.AGENT: (AgentMetadata, 'agentMetadata'),
            TraceType.CHAIN: (ChainMetadata, 'chainMetadata'),
            TraceType.EMBEDDING: (LlmMetadata, 'llmMetadata'),
            TraceType.LLM: (LlmMetadata, 'llmMetadata'),
            TraceType.QUERY: (QueryMetadata, 'queryMetadata'),
            TraceType.RETRIEVER: (RetrieverMetadata, 'retrieverMetadata'),
            TraceType.RERANKING: (RerankingMetadata, 'rerankingMetadata'),
            TraceType.SYNTHESIZE: (SynthesizeMetadata, 'synthesizeMetadata'),
            TraceType.TOOL: (ToolMetadata, 'toolMetadata'),
        }
        metadata_class, attribute_name = trace_mapping.get(self.trace_type, (None, None))
        if metadata_class and attribute_name:
            assert (isinstance(self.metadata, metadata_class), 
                    f"Metadata must be of type {metadata_class.__name__} for the {self.trace_type} trace type")
            setattr(current_trace, attribute_name, self.metadata)
        elif self.metadata is not None:
            assert (isinstance(self.metadata, GenericMetadata), 
                    f"Metadata must be of type GenericMetaData for Custom traces")

        # update track stack in trace manager
        trace_manager.set_trace_stack(current_stack)

    # change to attributes and custom attributes
    def set_parameters(self, metadata: Metadata):
        if self.trace_provider == TraceProvider.CUSTOM:
            assert (isinstance(metadata, GenericMetadata), 
                    f"Metadata must be of type GenericMetadata for CUSTOM Traces")
        self.metadata = metadata

    def track(
        self,
        event_name: str = None,
        model: str = None,
        input: str = None,
        response: str = None,
        retrieval_context: Optional[List[str]] = None,
        completion_time: Optional[float] = None,
        token_usage: Optional[float] = None,
        token_cost: Optional[float] = None,
        distinct_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        additional_data: Optional[Dict[str, str]] = None,
        hyperparameters: Optional[Dict[str, str]] = {},
        fail_silently: Optional[bool] = False,
        raise_exception: Optional[bool] = True,
        run_async: Optional[bool] = True,
    ):
        self.is_tracking = True
        self.track_params = {
            "event_name": event_name,
            "model": model,
            "input": input,
            "response": response,
            "retrieval_context": retrieval_context,
            "completion_time": completion_time,
            "token_usage": token_usage,
            "token_cost": token_cost,
            "distinct_id": distinct_id,
            "conversation_id": conversation_id,
            "additional_data": additional_data,
            "hyperparameters": hyperparameters,
            "fail_silently": fail_silently,
            "raise_exception": raise_exception,
            "run_async": run_async,
        }
