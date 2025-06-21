import json
from typing import Any

from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index_instrumentation.base import BaseEvent

from deepeval.tracing.types import BaseSpan, TraceSpanStatus
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
        