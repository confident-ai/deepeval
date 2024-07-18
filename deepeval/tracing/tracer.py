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
    LANGCHAIN = "LANGCHAIN"
    LLAMA_INDEX = "LLAMA_INDEX"
    DEFAULT = "DEFAULT"
    CUSTOM = "CUSTOM"
    HYBRID = "HYBRID"


class TraceType(Enum):
    AGENT = "Agent"
    CHAIN = "Chain"
    CHUNKING = "Chunking"
    EMBEDDING = "Embedding"
    LLM = "LLM"
    NODE_PARSING = "Node Parsing"
    QUERY = "Query"
    RERANKING = "Reranking"
    RETRIEVER = "Retriever"
    SYNTHESIZE = "Synthesize"
    TOOL = "Tool"


class LlamaIndexTraceType(Enum):
    AGENT = "Agent"
    CHAIN = "Chain"
    CHUNKING = "Chunking"
    EMBEDDING = "Embedding"
    LLM = "LLM"
    NODE_PARSING = "Node Parsing"
    QUERY = "Query"
    RERANKING = "Reranking"
    RETRIEVER = "Retriever"
    SYNTHESIZE = "Synthesize"
    TOOL = "Tool"


class LangChainTraceType(Enum):
    AGENT = "Agent"
    CHAIN = "Chain"
    CHUNKING = "Chunking"
    EMBEDDING = "Embedding"
    LLM = "LLM"
    NODE_PARSING = "Node Parsing"
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
    source_file: Optional[str] = Field(None, serialization_alias="sourceFile")


########################################################
### Attributes Types ###################################
########################################################


class AgentAttributes(BaseModel):
    input: str
    output: str
    name: str
    description: str


class ChainAttributes(BaseModel):
    input: str
    output: str
    # Optional variables
    prompt_template: Optional[str] = Field(
        None, serialization_alias="promptTemplate"
    )


class ChunkAttributes(BaseModel):
    input: str
    output_chunks: List[str] = Field([], serialization_alias="outputChunks")


class EmbeddingAttributes(BaseModel):
    embedding_text: str = Field("", serialization_alias="embeddingText")
    # Optional variables
    model: Optional[str] = None
    embedding_length: Optional[int] = Field(
        None, serialization_alias="embeddingLength"
    )


class LlmAttributes(BaseModel):
    input_str: str = Field("", serialization_alias="inputStr")
    output_str: str = Field("", serialization_alias="outputStr")
    # Optional variables
    model: Optional[str] = None
    total_token_count: Optional[int] = Field(
        None, serialization_alias="totalTokenCount"
    )
    prompt_token_count: Optional[int] = Field(
        None, serialization_alias="promptTokenCount"
    )
    completion_token_count: Optional[int] = Field(
        None, serialization_alias="completionTokenCount"
    )
    prompt_template: Optional[str] = Field(
        None, serialization_alias="promptTemplate"
    )
    prompt_template_variables: Optional[Dict[str, str]] = Field(
        None, serialization_alias="promptTemplateVariables"
    )


class NodeParsingAttributes(BaseModel):
    output_nodes: List[RetrievalNode] = Field(
        [], serialization_alias="outputNodes"
    )


class QueryAttributes(BaseModel):
    input: str
    output: str


class RetrieverAttributes(BaseModel):
    query_str: str = Field("", serialization_alias="queryStr")
    nodes: List[RetrievalNode]
    # Optional variables
    top_k: Optional[int] = Field(None, serialization_alias="topK")
    average_chunk_size: Optional[int] = Field(
        None, serialization_alias="averageChunkSize"
    )
    top_score: Optional[float] = Field(None, serialization_alias="topScore")
    similarity_scorer: Optional[str] = Field(
        None, serialization_alias="similarityScorer"
    )


class RerankingAttributes(BaseModel):
    input_nodes: List[RetrievalNode] = Field(
        [], serialization_alias="inputNodes"
    )
    output_nodes: List[RetrievalNode] = Field(
        [], serialization_alias="outputNodes"
    )
    # Optional variables
    model: Optional[str] = None
    top_n: Optional[int] = Field(None, serialization_alias="topN")
    batch_size: Optional[int] = Field(None, serialization_alias="batchSize")
    query_str: Optional[str] = Field(None, serialization_alias="queryStr")


class SynthesizeAttributes(BaseModel):
    user_query: str = Field("", serialization_alias="userQuery")
    response: str
    # Optional variables
    retrieved_context: Optional[str] = Field(
        None, serialization_alias="retrievedContext"
    )


class ToolAttributes(BaseModel):
    name: str
    description: str


class GenericAttributes(BaseModel):
    input: Optional[str] = None
    output: Optional[str] = None


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
    inputPayload: Optional[Dict]
    outputPayload: Optional[Dict]


@dataclass
class AgentTrace(BaseTrace):
    agentAttributes: AgentAttributes
    type: TraceType


@dataclass
class ChainTrace(BaseTrace):
    chainAttributes: ChainAttributes
    type: TraceType


@dataclass
class ChunkTrace(BaseTrace):
    chunkAttributes: ChunkAttributes
    type: TraceType


@dataclass
class EmbeddingTrace(BaseTrace):
    embeddingAttributes: EmbeddingAttributes
    type: TraceType


@dataclass
class LlmTrace(BaseTrace):
    llmAttributes: LlmAttributes
    type: TraceType


@dataclass
class NodeParsingTrace(BaseTrace):
    nodeParsingAttributes: NodeParsingAttributes
    type: TraceType


@dataclass
class QueryTrace(BaseTrace):
    queryAttributes: QueryAttributes
    type: TraceType


@dataclass
class RetrieverTrace(BaseTrace):
    retrieverAttributes: RetrieverAttributes
    type: TraceType


@dataclass
class RerankingTrace(BaseTrace):
    rerankingAttributes: RerankingAttributes
    type: TraceType


@dataclass
class SynthesizeTrace(BaseTrace):
    synthesizeAttributes: SynthesizeAttributes
    type: TraceType


@dataclass
class ToolTrace(BaseTrace):
    toolAttributes: ToolAttributes
    type: TraceType


@dataclass
class GenericTrace(BaseTrace):
    genericAttributes: Optional[GenericAttributes] = None
    type: str


Attributes = Union[
    AgentAttributes,
    ChainAttributes,
    EmbeddingAttributes,
    LlmAttributes,
    QueryAttributes,
    RerankingAttributes,
    RetrieverAttributes,
    SynthesizeAttributes,
    ToolAttributes,
    GenericAttributes,
]
TraceData = Union[
    AgentTrace,
    ChainTrace,
    EmbeddingTrace,
    LlmTrace,
    QueryTrace,
    RerankingTrace,
    RetrieverTrace,
    SynthesizeTrace,
    ToolTrace,
    GenericTrace,
]
TraceStack = List[TraceData]

# Context variable to maintain an isolated stack for each async task
trace_stack_var: ContextVar[TraceStack] = ContextVar("trace_stack", default=[])
dict_trace_stack_var: ContextVar[Dict] = ContextVar(
    "dict_trace_stack", default={}
)
outter_provider_var: ContextVar[Optional[TraceProvider]] = ContextVar(
    "outter_provider", default=None
)

########################################################
### ContextVar Managers ################################
########################################################


class TraceManager:

    def get_outter_provider(self):
        return outter_provider_var.get()

    def set_outter_provider(self, provider: TraceProvider):
        return outter_provider_var.set(provider)

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
        self.trace_type: TraceType | str = trace_type
        if isinstance(self.trace_type, TraceType):
            self.trace_provider = TraceProvider.DEFAULT
        else:
            self.trace_provider = TraceProvider.CUSTOM
        self.name: str
        self.start_time: float
        self.execution_time: float
        self.status: TraceStatus
        self.error: Optional[Dict[str:Any]] = None
        self.attributes: Optional[Attributes] = None
        self.track_params: Optional[Dict] = None
        self.is_tracking: bool = False

    def __enter__(self):
        # start timer
        self.start_time = perf_counter()

        # create name
        caller_frame = inspect.currentframe().f_back
        func_name = caller_frame.f_code.co_name
        self.name = func_name

        # set outtermost provider
        if not trace_manager.get_outter_provider():
            trace_manager.set_outter_provider(TraceProvider.CUSTOM)

        trace_instance: BaseTrace = self.create_trace_instance(
            self.trace_type, self.trace_provider, None
        )
        trace_manager.append_to_trace_stack(trace_instance)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Check attributes was set
        if not self.attributes and self.trace_provider == TraceProvider.DEFAULT:
            raise ValueError(
                f"`set_attributes` was not called before the end of a {self.trace_type} trace type."
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
        attributes: Optional[Attributes] = None,
    ):
        trace_kwargs = {
            "traceProvider": trace_provider,
            "type": trace_type,
            "executionTime": 0,
            "name": self.name,
            "status": TraceStatus.SUCCESS,
            "traces": [],
            "inputPayload": None,
            "outputPayload": None,
        }
        if trace_provider == TraceProvider.DEFAULT:
            if trace_type == TraceType.AGENT:
                trace_kwargs["agentAttributes"] = attributes
                return AgentTrace(**trace_kwargs)
            elif trace_type == TraceType.CHAIN:
                trace_kwargs["chainAttributes"] = attributes
                return ChainTrace(**trace_kwargs)
            elif trace_type == TraceType.EMBEDDING:
                trace_kwargs["embeddingAttributes"] = attributes
                return EmbeddingTrace(**trace_kwargs)
            elif trace_type == TraceType.LLM:
                trace_kwargs["llmAttributes"] = attributes
                return LlmTrace(**trace_kwargs)
            elif trace_type == TraceType.QUERY:
                trace_kwargs["queryAttributes"] = attributes
                return QueryTrace(**trace_kwargs)
            elif trace_type == TraceType.RERANKING:
                trace_kwargs["rerankingAttributes"] = attributes
                return RerankingTrace(**trace_kwargs)
            elif trace_type == TraceType.RETRIEVER:
                trace_kwargs["retrieverAttributes"] = attributes
                return RetrieverTrace(**trace_kwargs)
            elif trace_type == TraceType.SYNTHESIZE:
                trace_kwargs["synthesizeAttributes"] = attributes
                return SynthesizeTrace(**trace_kwargs)

        elif trace_provider == TraceProvider.CUSTOM:
            trace_kwargs["genericAttributes"] = attributes
            return GenericTrace(**trace_kwargs)

    def update_trace_instance(self):
        # Get the latest trace instance from the stack
        current_stack = trace_manager.get_trace_stack_copy()
        current_trace = current_stack[-1]

        # update current_trace
        current_trace.executionTime = self.execution_time
        current_trace.status = self.status

        # Assert that the attributes is of the correct type and assign it to the current trace
        trace_mapping = {
            TraceType.AGENT: (AgentAttributes, "agentAttributes"),
            TraceType.CHAIN: (ChainAttributes, "chainAttributes"),
            TraceType.EMBEDDING: (EmbeddingAttributes, "embeddingAttributes"),
            TraceType.LLM: (LlmAttributes, "llmAttributes"),
            TraceType.QUERY: (QueryAttributes, "queryAttributes"),
            TraceType.RETRIEVER: (RetrieverAttributes, "retrieverAttributes"),
            TraceType.RERANKING: (RerankingAttributes, "rerankingAttributes"),
            TraceType.SYNTHESIZE: (
                SynthesizeAttributes,
                "synthesizeAttributes",
            ),
            TraceType.TOOL: (ToolAttributes, "toolAttributes"),
        }
        attribute_class, attribute_name = trace_mapping.get(
            self.trace_type, (None, None)
        )
        if attribute_class and attribute_name:
            assert isinstance(
                self.attributes, attribute_class
            ), f"Attributes must be of type {attribute_class.__name__} for the {self.trace_type} trace type"
            setattr(current_trace, attribute_name, self.attributes)
        if self.trace_provider == TraceProvider.CUSTOM and self.attributes:
            setattr(current_trace, "genericAttributes", self.attributes)

        # update track stack in trace manager
        trace_manager.set_trace_stack(current_stack)

    # change to attributes and custom attributes
    def set_attributes(self, attributes: Attributes):
        if self.trace_provider == TraceProvider.CUSTOM:
            assert (
                isinstance(attributes, GenericAttributes),
                f"Attributes must be of type GenericAttributes for CUSTOM Traces",
            )

        # append trace instance to stack
        self.attributes = attributes

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
