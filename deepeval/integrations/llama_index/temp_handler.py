from typing import Any, Dict, Optional
import inspect
from time import perf_counter

import deepeval
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.types import BaseSpan, TraceSpanStatus
try:
    from llama_index.core.instrumentation.events.base import BaseEvent
    from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
    from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
    from llama_index.core.instrumentation.span.base import BaseSpan as LlamaIndexBaseSpan
    import llama_index.core.instrumentation as instrument
    llama_index_installed = True
except:
    llama_index_installed = False

def is_llama_index_installed():
    if not llama_index_installed:
        raise ImportError("llama-index is neccesary for this functionality. Please install it with `pip install llama-index` or with package manager of choice.")


class LLamaIndexHandler(BaseEventHandler, BaseSpanHandler):
    active_trace_uuid: Optional[str] = None
    
    def __init__(self):
        capture_tracing_integration("llama-index")
        is_llama_index_installed()
        super().__init__()

    def handle(self, event: BaseEvent, **kwargs) -> Any:
        pass
    
    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[LlamaIndexBaseSpan]:
        if parent_span_id is None:
            self.active_trace_uuid = trace_manager.start_new_trace().uuid

        base_span = BaseSpan(
            uuid=id_,
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_uuid,
            parent_uuid=parent_span_id,
            start_time=perf_counter(),
            name="llama-index", # add the name of the class
            input=None, # add the input 
        )

        trace_manager.add_span(base_span)
        trace_manager.add_span_to_trace(base_span)
        
        return base_span

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[LlamaIndexBaseSpan]:
        base_span = trace_manager.get_span_by_uuid(id_)
        if base_span is None:
            return None
        
        base_span.end_time = perf_counter()
        base_span.status = TraceSpanStatus.SUCCESS
        base_span.output = None # add the output
        trace_manager.remove_span(base_span.uuid)

        if base_span.parent_uuid is None:
            trace_manager.end_trace(base_span.trace_uuid)
            self.active_trace_uuid = None

        return base_span
    
    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[LlamaIndexBaseSpan]:
        base_span = trace_manager.get_span_by_uuid(id_)
        if base_span is None:
            return None
        
        base_span.end_time = perf_counter()
        base_span.status = TraceSpanStatus.SUCCESS # find a way to add error and handle the span without the parent id

        if base_span.parent_uuid is None:
            trace_manager.end_trace(base_span.trace_uuid)
            self.active_trace_uuid = None

        return base_span


def instrumentator(api_key: Optional[str] = None):
    if api_key:
        deepeval.login_with_confident_api_key(api_key)
    
    handler = LLamaIndexHandler()
    
    dispatcher = instrument.get_dispatcher()
    dispatcher.add_event_handler(handler)
    dispatcher.add_span_handler(handler)
    return None