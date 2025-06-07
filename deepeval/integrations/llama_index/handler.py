from datetime import datetime, timezone
import json
from typing import Any
from llama_index.core.instrumentation.event_handlers.base import (
    BaseEventHandler,
    BaseEvent
)

from deepeval.tracing.api import BaseApiSpan, SpanApiType, TraceApi, TraceSpanApiStatus
from deepeval.tracing.utils import to_zod_compatible_iso
from deepeval.tracing import trace_manager

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
        _event_dict = event.dict()
        _fallback_json = json.loads(json.dumps(_event_dict, default=serialize, indent=4))

        _base_api_span = BaseApiSpan(
            uuid=_event_dict["id_"],
            name=_event_dict["class_name"],
            status=TraceSpanApiStatus.SUCCESS,
            type=SpanApiType.BASE,
            traceUuid=_event_dict["id_"],
            startTime=to_zod_compatible_iso(datetime.fromtimestamp(int(_event_dict["timestamp"].timestamp()) / 1e9, tz=timezone.utc)),
            endTime=to_zod_compatible_iso(datetime.fromtimestamp(int(_event_dict["timestamp"].timestamp()) / 1e9, tz=timezone.utc)),
            #metadata=_fallback_json
        )

        _trace_api = TraceApi(
            uuid=_event_dict["id_"]+"-llama-index",
            baseSpans=[_base_api_span],
            agentSpans=[],
            llmSpans=[],
            retrieverSpans=[],
            toolSpans=[],
            startTime=to_zod_compatible_iso(datetime.fromtimestamp(int(_event_dict["timestamp"].timestamp()) / 1e9, tz=timezone.utc)),
            endTime=to_zod_compatible_iso(datetime.fromtimestamp(int(_event_dict["timestamp"].timestamp()) / 1e9, tz=timezone.utc)),
            #metadata=_fallback_json
            environment="development"
        )
        trace_manager.post_trace_api(_trace_api)

my_event_handler = LLamaIndexEventHandler()