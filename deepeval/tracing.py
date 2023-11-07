from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, List, Union, Optional
from time import perf_counter
import traceback
from inspect import signature
import threading
from deepeval.utils import dataclass_to_dict


class TraceType(Enum):
    LLM = "LLM"
    RETRIEVER = "Retriever"
    EMBEDDING = "Embedding"
    TOOL = "Tool"
    AGENT = "Agent"
    CHAIN = "Chain"


class TraceStatus(Enum):
    SUCCESS = "Success"
    ERROR = "Error"


@dataclass
class LlmMetadata:
    model: str


@dataclass
class EmbeddingMetadata:
    model: str


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
    input: str
    llmMetadata: LlmMetadata = None


@dataclass
class EmbeddingTrace(BaseTrace):
    embeddingMetadata: EmbeddingMetadata


@dataclass
class GenericTrace(BaseTrace):
    type: str


TraceData = Union[LlmTrace, EmbeddingTrace, GenericTrace]


class TraceManager:
    def __init__(self):
        self._local = threading.local()

    def get_trace_stack(self):
        if not hasattr(self._local, "trace_stack"):
            self._local.trace_stack = []
            self._local.dict_trace_stack = None
        return self._local.trace_stack

    def clear_trace_stack(self):
        self.get_trace_stack().clear()

    def pop_trace_stack(self):
        if self.get_trace_stack():
            self.get_trace_stack().pop()

    def append_to_trace_stack(self, trace_instance):
        self.get_trace_stack().append(trace_instance)

    def set_dict_trace_stack(self, dict_trace_stack):
        self._local.dict_trace_stack = dict_trace_stack

    def get_and_reset_dict_trace_stack(self):
        dict_trace_stack = getattr(self._local, "dict_trace_stack", None)
        self._local.dict_trace_stack = None
        return dict_trace_stack


trace_manager = TraceManager()


def trace(
    type: str,
    name: Optional[str] = None,
    model: Optional[str] = None,
):
    assert isinstance(
        type, Union[TraceType, str]
    ), "'type' must be a 'TraceType' or str"

    if type in [TraceType.LLM, TraceType.EMBEDDING] and model is None:
        raise ValueError(f"{type} trace type requires a model.")
    assert model is None or isinstance(
        model, str
    ), "'model' must be a str or None"

    if type not in [TraceType.LLM, TraceType.EMBEDDING] and model is not None:
        raise ValueError(
            f"Parameter 'model' should not be provided for {type} trace types."
        )

    def decorator_trace(func: Callable):
        if type == TraceType.LLM:
            sig = signature(func)
            params = sig.parameters.values()

            # Check if it's an instance method, adjust parameter list if 'self' or 'cls' is present
            if any(p.name in ["self", "cls"] for p in params):
                params = [p for p in params if p.name not in ["self", "cls"]]

            # There should be exactly one parameter left of type list[str]
            if len(params) != 1:
                raise ValueError(
                    "Function of type `TraceType.LLM` must have exactly one parameter of type 'list[str]'"
                )

        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            if type == TraceType.LLM:
                input_str = (
                    args[1]
                    if "self" in sig.parameters or "cls" in sig.parameters
                    else args[0]
                )
                if not isinstance(input_str, str):
                    raise ValueError(
                        "Argument type for `TraceType.LLM` must be a string"
                    )

            bound_method = False
            # Check if it is called with 'self' or 'cls' parameter
            params = sig.parameters
            if args:
                first_param = next(iter(params))
                if first_param == "self" or first_param == "cls":
                    bound_method = True

            # Remove 'self' or 'cls' parameter if function is a method
            if bound_method:
                trace_args = args[1:]
            else:
                trace_args = args

            # Proceed to create your trace, using trace_args instead of args
            trace_instance_input = {"args": trace_args, "kwargs": kwargs}

            trace_instance = None
            effective_name = name if name is not None else func.__name__
            if type == TraceType.LLM:
                trace_instance = LlmTrace(
                    type=type,
                    executionTime=0,
                    name=effective_name,
                    input=input_str,
                    output=None,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    llmMetadata=LlmMetadata(model=model),
                )
            elif type == TraceType.EMBEDDING:
                trace_instance = EmbeddingTrace(
                    type=type,
                    executionTime=0,
                    name=effective_name,
                    input=trace_instance_input,
                    output=None,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                    embeddingMetadata=EmbeddingMetadata(model=model),
                )
            else:
                trace_instance = GenericTrace(
                    type=type,
                    executionTime=0,
                    name=effective_name,
                    input=trace_instance_input,
                    output=None,
                    status=TraceStatus.SUCCESS,
                    traces=[],
                )

            trace_manager.append_to_trace_stack(trace_instance)
            start_time = perf_counter()
            try:
                result = func(*args, **kwargs)
                trace_instance.output = result

            except Exception as e:
                trace_instance.status = TraceStatus.ERROR
                trace_instance.output = {
                    "type": __builtins__["type"](e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                raise e

            finally:
                trace_instance.executionTime = perf_counter() - start_time

                current_trace_stack = trace_manager.get_trace_stack()
                if len(current_trace_stack) > 1:
                    parent_trace = current_trace_stack[-2]
                    parent_trace.traces.append(trace_instance)

                if len(current_trace_stack) == 1:
                    dict_representation = dataclass_to_dict(
                        current_trace_stack[0]
                    )
                    trace_manager.set_dict_trace_stack(dict_representation)
                    trace_manager.clear_trace_stack()
                else:
                    trace_manager.pop_trace_stack()

            return result

        return wrapper

    return decorator_trace


def set_token_usage(tokens: int):
    pass


def get_trace_stack():
    return trace_manager.get_and_reset_dict_trace_stack()
