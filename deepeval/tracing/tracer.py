from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Union, Optional, Dict
from time import perf_counter
from pydantic import BaseModel, Field
import inspect

from deepeval.utils import dataclass_to_dict
from deepeval.prompt import Prompt
from deepeval.test_case import ToolCall


class Provider(Enum):
    LANGCHAIN = "LANGCHAIN"
    LLAMA_INDEX = "LLAMA_INDEX"
    DEFAULT = "DEFAULT"
    CUSTOM = "CUSTOM"
    HYBRID = "HYBRID"


class SpanType(Enum):
    AGENT = "Agent"
    LLM = "LLM"
    RETRIEVER = "Retriever"
    TOOL = "Tool"


class TraceSpanStatus(Enum):
    SUCCESS = "Success"
    ERROR = "Error"


class AgentAttributes(BaseModel):
    # input
    input: str
    # output
    output: str


class LlmAttributes(BaseModel):
    # input
    input: str
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



class RetrieverAttributes(BaseModel):
    # input
    embedding_text: str = Field(serialization_alias="embeddingText")
    # output
    retrieval_context : List[str]

    # Optional variables
    top_k: Optional[int] = Field(None, serialization_alias="topK")
    chunk_size: Optional[int] = Field(None, serialization_alias="chunkSize")


# Don't have to call this manually
class ToolAttributes(BaseModel):
    # input
    input_parameters: Optional[Dict[str, Any]] = Field(
        None, serialization_alias="inputParameters"
    )
    # output
    output: Optional[Any] = None



########################################################
### Trace Types #######################################
########################################################


@dataclass
class BaseSpan:
    uuid: str
    trace_uuid: str = Field(serialization_alias="traceUuid")
    parent_uuid: Optional[str] = Field(None, serialization_alias="parentUuid")
    start_time : float = Field(serialization_alias="startTime")
    end_time : Union[float, None] = Field(None, serialization_alias="endTime")
    status: TraceSpanStatus
    provider: Provider
    children: List["BaseSpan"]
    name: Optional[str] = None
    metadata: Optional[Dict] = None

    # Late populate
    input: Union[str, Dict, list, None]
    output: Union[str, Dict, list, None]


@dataclass
class AgentSpan(BaseSpan):
    name: str
    available_tools: List[str]
    attributes: AgentAttributes


@dataclass
class LlmSpan(BaseSpan):
    model: str
    cost_per_input_token: Optional[float] = Field(None, serialization_alias="costPerInputToken")
    cost_per_output_token: Optional[float] = Field(None, serialization_alias="costPerOutputToken")
    attributes: LlmAttributes


@dataclass
class RetrieverSpan(BaseSpan):
    embedder: str
    attributes: RetrieverAttributes


@dataclass
class ToolSpan(BaseSpan):
    name: str
    description: Optional[str] = None
    attributes: ToolAttributes


Attributes = Union[
    AgentAttributes,
    LlmAttributes,
    RetrieverAttributes,
    ToolAttributes
]

SpanType = Union[
    BaseSpan,
    AgentSpan,
    LlmSpan,
    RetrieverSpan,
    ToolSpan,
]

class Trace:
    spans: List[BaseSpan]
    status: TraceSpanStatus
    start_time : float = Field(serialization_alias="startTime")
    end_time : Union[float, None] = Field(None, serialization_alias="endTime")
    metadata: Optional[Dict] = None
    name: Optional[str] = None

# Context variable to maintain an isolated stack for each async task
trace_stack_var: ContextVar[Trace] = ContextVar("trace_stack", default=[])
track_params_var: ContextVar[Dict] = ContextVar("track_params", default={})
dict_trace_stack_var: ContextVar[Dict] = ContextVar(
    "dict_trace_stack", default={}
)
outter_provider_var: ContextVar[Optional[Provider]] = ContextVar(
    "outter_provider", default=None
)


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

    def set_track_params(self, track_params):
        track_params_var.set(track_params)

    def get_track_params(self):
        return track_params_var.get()

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
        else:
            trace_manager.pop_trace_stack()

    def create_span_instance(
        self,
        provider: Provider,
        attributes: Optional[Attributes] = None,
    ):
        trace_kwargs = {
            "traceProvider": trace_provider,
            "type": trace_type,
            "executionTime": 0,
            "name": self.name,
            "status": SpanStatus.SUCCESS,
            "traces": [],
            "inputPayload": None,
            "outputPayload": None,
            "parentId": None,
            "rootParentId": None,
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

    def update_span_instance(self):
        # Get the latest trace instance from the stack
        current_stack = trace_manager.get_trace_stack_copy()
        current_trace = current_stack[-1]

        # update current_trace
        current_trace.executionTime = self.execution_time
        current_trace.status = self.status

        # Assert that the attributes is of the correct type and assign it to the current trace
        span_mapping = {
            SpanType.AGENT: (AgentAttributes, "agentAttributes"),
            SpanType.LLM: (LlmAttributes, "llmAttributes"),
            SpanType.RETRIEVER: (RetrieverAttributes, "retrieverAttributes"),
            SpanType.TOOL: (ToolAttributes, "toolAttributes"),
        }
        attribute_class, attribute_name = span_mapping.get(
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
            assert isinstance(
                attributes, GenericAttributes
            ), f"Attributes must be of type GenericAttributes for CUSTOM Traces"

        # append trace instance to stack
        self.attributes = attributes
