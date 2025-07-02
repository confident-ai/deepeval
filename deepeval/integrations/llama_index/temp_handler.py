from typing import Any, Dict, Optional
import inspect
import deepeval
try:
    from llama_index.core.instrumentation.events.base import BaseEvent
    from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
    from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
    from llama_index.core.instrumentation.span.base import BaseSpan
    import llama_index.core.instrumentation as instrument
    llama_index_installed = True
except:
    llama_index_installed = False
    

def is_llama_index_installed():
    if not llama_index_installed:
        raise ImportError("llama-index is neccesary for this functionality. Please install it with `pip install llama-index` or with package manager of choice.")


class LLamaIndexEventHandler(BaseEventHandler):
    def handle(self, event: BaseEvent, **kwargs) -> Any:
        print("----handle----")

class LLamaIndexSpanHandler(BaseSpanHandler):
    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[BaseSpan]:
        print("----new_span----")

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[BaseSpan]:
        print("----prepare_to_exit_span----")

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[BaseSpan]:
        print("----prepare_to_drop_span----")


class LLamaIndexHandler():
    def __init__(self):
        
        is_llama_index_installed()
        self.event_handler = LLamaIndexEventHandler()
        self.span_handler = LLamaIndexSpanHandler()
        
        self.dispatcher = instrument.get_dispatcher()
        self.dispatcher.add_event_handler(self.event_handler)
        self.dispatcher.add_span_handler(self.span_handler)


def instrumentator(api_key: Optional[str] = None):
    if api_key:
        deepeval.login_with_confident_api_key(api_key)
    
    LLamaIndexHandler()
    return None