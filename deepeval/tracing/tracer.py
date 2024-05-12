from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Union, Optional, Dict
from time import perf_counter
import inspect
import time

from deepeval.utils import dataclass_to_dict
from deepeval.event import track

########################################################
### Trace Types ########################################
########################################################

class TraceType(Enum):
    LLM = "LLM"
    RETRIEVER = "Retriever"
    EMBEDDING = "Embedding"
    TOOL = "Tool"
    AGENT = "Agent"
    AGENT_STEP = "Agent Step"
    CHAIN = "Chain"
    CHUNKING = "Chunking"
    NODE_PARSING = "Node Parsing"
    SYNTHESIZE = "Synthesize"
    QUERY = "Query"
    RERANKING = "Reranking"


class TraceStatus(Enum):
    SUCCESS = "Success"
    ERROR = "Error"


@dataclass
class LlmMetadata:
    model: Optional[str] = None
    tokenCount: Optional[Dict[str, int]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    outputMessages: Optional[List[Dict[str, str]]] = None
    llmPromptTemplate: Optional[Any] = None
    llmPromptTemplateVariables: Optional[Any] = None


@dataclass
class EmbeddingMetadata:
    model: Optional[str] = None


@dataclass
class BaseTrace:
    type: Union[TraceType, str]
    executionTime: float
    name: str
    input: dict
    output: dict
    status: TraceStatus
    traces: List["TraceData"]


@dataclass
class LlmTrace(BaseTrace):
    llmMetadata: LlmMetadata


@dataclass
class EmbeddingTrace(BaseTrace):
    embeddingMetadata: EmbeddingMetadata


@dataclass
class GenericTrace(BaseTrace):
    type: str

Metadata = Union[EmbeddingMetadata, LlmMetadata]
TraceData = Union[LlmTrace, EmbeddingTrace, GenericTrace]
TraceStack = List[TraceData]

# Context variable to maintain an isolated stack for each async task
trace_stack_var: ContextVar[TraceStack] = ContextVar('trace_stack', default=[])
dict_trace_stack_var = ContextVar("dict_trace_stack", default=None)

########################################################
### Trace Manager ######################################
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
    def __init__(self, trace_type: str, **params: Optional[Dict]):
        self.trace_type = trace_type
        self.params = params
        self.start_time = None
        self.execution_time = None
        self.output = None
        self.metadata = None
        self.track_params: Optional[Dict] = None
        self.status_results = {}

    def __enter__(self):
        # start timer
        self.start_time = perf_counter()

        # create input
        caller_frame = inspect.currentframe().f_back
        args, _, _, locals_ = inspect.getargvalues(caller_frame)
        self.params['input'] = {arg: locals_[arg] for arg in args if arg not in ['self', 'cls']}

        # create name
        func_name = caller_frame.f_code.co_name 
        self.params['name'] = func_name

        # append trace instance to stack
        trace_instance = self.create_trace_instance(self.trace_type, **self.params)
        trace_manager.append_to_trace_stack(trace_instance)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop the span timing and calculate execution time
        self.execution_time = perf_counter() - self.start_time

        # Check if an exception occurred within the `with` block
        if exc_type is not None:
            self.status_results = {
                'status': 'ERROR',
                'exception_type': exc_type.__name__,
                'message': str(exc_val)
            }
        else:
            self.status_results['status'] = 'SUCCESS'

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

            # # send event to deepeval
            # track(
            #     event_name=self.track_params['event_name'],
            #     model=self.track_params['model'],
            #     input=self.track_params['input'],
            #     response=self.track_params['response'],
            #     retrieval_context=self.track_params['retrieval_context'],
            #     completion_time=self.track_params['completion_time'],
            #     token_usage=self.track_params['token_usage'],
            #     token_cost=self.track_params['token_cost'],
            #     distinct_id=self.track_params['distinct_id'],
            #     conversation_id=self.track_params['conversation_id'],
            #     additional_data=self.track_params['additional_data'],
            #     hyperparameters=self.track_params['hyperparameters'],
            #     fail_silently=self.track_params['fail_silently'],
            #     raise_expection=self.track_params['raise_expection'],
            #     run_async=self.track_params['run_async'],
            # )

            
        else:
            trace_manager.pop_trace_stack()
    
    def create_trace_instance(self, trace_type, **params):
        if trace_type == TraceType.LLM:
            return LlmTrace(
                type=trace_type, 
                executionTime=0, 
                name=params.get('name', ''), 
                input=params.get('input', None), 
                output=None, 
                status=TraceStatus.SUCCESS, 
                traces=[], 
                llmMetadata=params.get('llmMetadata', None))
        
        elif trace_type == TraceType.EMBEDDING:
            return EmbeddingTrace(
                type=trace_type, 
                executionTime=0, 
                ame=params.get('name', ''), 
                input=params.get('input', None), 
                output=None, 
                status=TraceStatus.SUCCESS, 
                traces=[], 
                embeddingMetadata=params.get('embeddingMetadata', None))
        else:
            return GenericTrace(
                type=trace_type, 
                executionTime=0, 
                name=params.get('name', ''), 
                input=params.get('input', None),
                output=None, 
                status=TraceStatus.SUCCESS, 
                traces=[])
        
    def update_trace_instance(self):
        # Get the latest trace instance from the stack
        current_stack = trace_manager.get_trace_stack_copy()
        current_trace = current_stack[-1]

        # update current_trace
        current_trace.executionTime = self.execution_time
        current_trace.status = self.status_results['status']
        current_trace.output = self.output

        # Update metadata in current_trace
        if self.trace_type == TraceType.LLM:
            assert isinstance(self.metadata, LlmMetadata), "Metadata must be of type LlmMetadata for LLM trace type"
            current_trace.llmMetadata=self.metadata

        elif self.trace_type == TraceType.EMBEDDING:
            assert isinstance(self.metadata, EmbeddingMetadata), "Metadata must be of type EmbeddingMetadata for EMBEDDING trace type"
            current_trace.embeddingMetadata=self.metadata

        trace_manager.set_trace_stack(current_stack)
    
    def set_parameters(self, output: Any, metadata: Optional[Metadata]=None):
        self.output = output

        if not metadata and self.trace_type == TraceType.LLM:
            self.metadata = LlmMetadata()
        elif not metadata and self.trace_type == TraceType.EMBEDDING:
            self.metadata = EmbeddingMetadata()
    
    def track(
        self,
        event_name: Optional[str] = None,
        model: Optional[str] = None,
        input: Optional[str] = None,
        response: Optional[str] = None,
        retrieval_context: Optional[List[str]] = None,
        completion_time: Optional[float] = None,
        token_usage: Optional[float] = None,
        token_cost: Optional[float] = None,
        distinct_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        additional_data: Optional[Dict[str, str]] = None,
        hyperparameters: Optional[Dict[str, str]] = {},
        fail_silently: Optional[bool] = False,
        raise_expection: Optional[bool] = True,
        run_async: Optional[bool] = True,
    ):
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
        "raise_exception": raise_expection,
        "run_async": run_async
    }
