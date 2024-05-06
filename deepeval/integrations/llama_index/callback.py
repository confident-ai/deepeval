from typing import Any, Dict, List, Optional, Union
from time import perf_counter

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms import ChatMessage

from deepeval.tracing import (
    trace_manager,
    get_trace_stack,
    LlmTrace,
    GenericTrace,
    EmbeddingTrace,
    TraceStatus,
    LlmMetadata,
    EmbeddingMetadata,
    TraceType,
)
from deepeval.utils import dataclass_to_dict

events_to_ignore = [
    CBEventType.CHUNKING,
    CBEventType.NODE_PARSING,
    CBEventType.EMBEDDING,
    CBEventType.TREE,
    CBEventType.SUB_QUESTION,
    CBEventType.FUNCTION_CALL,
    CBEventType.EXCEPTION,
    CBEventType.AGENT_STEP,
]


class LlamaIndexCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.event_map = {}
        super().__init__(
            event_starts_to_ignore=events_to_ignore,
            event_ends_to_ignore=events_to_ignore,
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self.event_map = {}
        trace_manager.clear_trace_stack()
        return

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        return

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        trace_instance = self.create_trace_instance(event_type)
        self.event_map[event_id] = trace_instance
        trace_manager.append_to_trace_stack(trace_instance)
        return

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        trace_instance = self.event_map[event_id]
        trace_instance.executionTime = (
            perf_counter() - trace_instance.executionTime
        )
        input_kwargs = {}
        if payload is not None:
            for event in EventPayload:
                value = payload.get(event.value)
                if value is not None:
                    input_kwargs[event.value] = value

        current_trace_stack = trace_manager.get_trace_stack()
        if len(current_trace_stack) > 1:
            parent_trace = current_trace_stack[-2]
            parent_trace.traces.append(trace_instance)

        if len(current_trace_stack) == 1:
            dict_representation = dataclass_to_dict(current_trace_stack[0])
            trace_manager.set_dict_trace_stack(dict_representation)
            trace_manager.clear_trace_stack()
        else:
            trace_manager.pop_trace_stack()

        return

    def create_trace_instance(
        self, event_type: CBEventType
    ) -> Union[EmbeddingTrace, LlmMetadata, GenericTrace]:
        current_time = perf_counter()
        type = self.convert_event_type_to_deepeval_trace_type(event_type)
        name = event_type.capitalize()
        trace_instance_input = {"args": None, "kwargs": None}
        if event_type == CBEventType.LLM:
            trace_instance = LlmTrace(
                type=type,
                executionTime=current_time,
                name=name,
                input=trace_instance_input,
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
                llmMetadata=LlmMetadata(model="None"),
            )
        elif event_type == CBEventType.EMBEDDING:
            trace_instance = EmbeddingTrace(
                type=type,
                executionTime=current_time,
                name=name,
                input=trace_instance_input,
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
                embeddingMetadata=EmbeddingMetadata(model="None"),
            )
        else:
            trace_instance = GenericTrace(
                type=type,
                executionTime=current_time,
                name=name,
                input=trace_instance_input,
                output=None,
                status=TraceStatus.SUCCESS,
                traces=[],
            )

        return trace_instance

    def convert_event_type_to_deepeval_trace_type(
        self, event_type: CBEventType
    ):
        # TODO: add more types
        if event_type == CBEventType.LLM:
            return TraceType.LLM
        elif event_type == CBEventType.RETRIEVE:
            return TraceType.RETRIEVER
        elif event_type == CBEventType.EMBEDDING:
            return TraceType.EMBEDDING

        return event_type.value.capitalize()
