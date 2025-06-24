from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any, Optional
from uuid import UUID

from deepeval.tracing import trace_manager

class CallbackHandler(BaseCallbackHandler):

    active_trace_id: Optional[str] = None

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            if self.active_trace_id is None:
                self.active_trace_id = trace_manager.start_new_trace().uuid

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            trace_manager.end_trace(self.active_trace_id)
            self.active_trace_id = None
