import datetime
from typing import Any, Dict, List, Optional, Union, cast

from time import perf_counter
from llama_index.bridge.pydantic import BaseModel
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.llms import ChatMessage

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
        print("trace starting")
        self.event_map = {}
        return

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        print("trace ending")
        return

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        # make this a trace object instead, empty kwargs
        self.event_map[event_id] = perf_counter()

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        self.event_map[event_id] = perf_counter() - self.event_map[event_id]
        # make fill in trace object in kwargs based on CBEventType

        return
