import inspect
import json
import time
from time import perf_counter
from typing import Any, Dict, Optional, TypeVar

from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index_instrumentation.base import BaseEvent

from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from llama_index.core.instrumentation.span.base import BaseSpan

T = TypeVar("T", bound=BaseSpan)

from deepeval.tracing.types import BaseSpan, Trace, TraceSpanStatus
from deepeval.tracing import trace_manager
from deepeval.tracing.context import current_span_context

def serialize(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)  # fallback

class LLamaIndexEventHandler(BaseEventHandler):
    """LlamaIndex custom EventHandler."""

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LLamaIndexEventHandler"

    def handle(self, event: BaseEvent, **kwargs) -> Any:
        """Logic for handling event."""
        
        # Get the current span from the context
        parent_span = current_span_context.get()

        # mandatory fields
        parent_uuid = None
        trace_uuid = None

        # Determine trace_uuid and parent_uuid before creating the span instance
        if parent_span:
            parent_uuid = parent_span.uuid
            trace_uuid = parent_span.trace_uuid
        
        # Generate UUIDs if they are not available
        if parent_uuid is None or trace_uuid is None:
            # TODO: Implement for more use cases where parent_uuid and trace_uuid are not available
            pass

        event_dict = event.dict()
        
        # TODO: Map the values with relevant attributes
        fallback_json = json.loads(json.dumps(event_dict, default=serialize, indent=4))
        
        base_span = BaseSpan(
            uuid=event_dict["id_"],
            status=TraceSpanStatus.SUCCESS, # TODO: Add more statuses
            children=[],
            trace_uuid=trace_uuid,
            parent_uuid=parent_uuid,
            start_time=event_dict["timestamp"].timestamp(),
            end_time=event_dict["timestamp"].timestamp(),
            name=event_dict["class_name"],
            metadata=fallback_json
        )

        trace_manager.add_span_to_trace(base_span)

class LLamaIndexSpanHandler(BaseSpanHandler):
    """LlamaIndex custom SpanHandler."""

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """
        Create a span.

        Subclasses of BaseSpanHandler should create the respective span type T
        and return it. Only NullSpanHandler should return a None here.
        """
        # prepare fallback
        new_span_fallback_json = {
            "id_": id_,
            "bound_args": serialize(bound_args),
            "instance": serialize(instance),
            "parent_span_id": parent_span_id,
            "tags": serialize(tags),
            "kwargs": serialize(kwargs),
        }

        # if parent_span_id is present, add the span to the parent trace
        if parent_span_id:
            base_span = BaseSpan(
                uuid=id_,
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=parent_span_id,
                parent_uuid=parent_span_id,
                start_time=perf_counter(),
                name=id_,
                metadata={"new_span_fallback_json": new_span_fallback_json}
            )
            trace_manager.add_span(base_span)
            trace_manager.add_span_to_trace(base_span)
        else:
            # if parent_span_id is not present, create a new trace
            new_trace = Trace(
                uuid=id_,
                root_spans=[],
                status=TraceSpanStatus.IN_PROGRESS,
                start_time=time.time(),
                confident_api_key=trace_manager.confident_api_key,
            )
            trace_manager.active_traces[id_] = new_trace
            trace_manager.traces.append(new_trace)
            
            # create a new span
            base_span = BaseSpan(
                uuid=id_,
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=id_,
                parent_uuid=None,
                start_time=perf_counter(),
                name=id_,
                metadata={"new_span_fallback_json": new_span_fallback_json}
            )
            trace_manager.add_span(base_span)
            trace_manager.add_span_to_trace(base_span)
    
    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """
        Logic for preparing to exit a span.

        Subclasses of BaseSpanHandler should return back the specific span T
        that is to be exited. If None is returned, then the span won't actually
        be exited.
        """
        
        # prepare fallback
        exit_span_fallback_json = {
            "id_": id_,
            "bound_args": serialize(bound_args),
            "instance": serialize(instance),
            "result": serialize(result),
            "kwargs": serialize(kwargs),
        }
        # fetch from the active_spans
        span = trace_manager.active_spans[id_]
        # update the fallback metadata
        span.end_time = perf_counter()
        span.metadata["exit_span_fallback_json"] = exit_span_fallback_json
        span.status = TraceSpanStatus.SUCCESS
        
        # remove the span from the active_spans
        trace_manager.remove_span(id_)
        
        # if root, end equivalent trace 
        if span.parent_uuid is None:
            trace = trace_manager.active_traces[span.trace_uuid]
            trace.status = TraceSpanStatus.SUCCESS
            trace_manager.end_trace(trace.uuid)
    
    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Logic for preparing to drop a span."""
        # prepare fallback
        exit_span_fallback_json = {
            "id_": id_,
            "bound_args": serialize(bound_args),
            "instance": serialize(instance),
            "err": serialize(err),
            "kwargs": serialize(kwargs),
        }
        # fetch from the active_spans
        span = trace_manager.active_spans[id_]
        # update the fallback metadata
        span.end_time = perf_counter()
        span.metadata["exit_span_fallback_json"] = exit_span_fallback_json
        span.status = TraceSpanStatus.ERRORED
        
        # remove the span from the active_spans
        trace_manager.remove_span(id_)
        
        # if root, end equivalent trace 
        if span.parent_uuid is None:
            trace = trace_manager.active_traces[span.trace_uuid]
            trace.end_time = perf_counter()
            trace.status = TraceSpanStatus.ERRORED
            trace_manager.end_trace(trace.uuid)
