from typing import Optional, List, Dict, Any
from contextvars import ContextVar
from contextlib import contextmanager

from .tracing import trace_manager
from .context import current_trace_context, update_current_trace

from deepeval.prompt import Prompt
current_prompt_context: ContextVar[Optional[Prompt]] = ContextVar(
    "current_prompt", default=None
)

@contextmanager
def trace(
    prompt: Optional[Prompt] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
):
    current_trace = current_trace_context.get()

    if not current_trace:
        current_trace = trace_manager.start_new_trace()
        
    current_trace_context.set(current_trace)
    
    # set the current prompt context
    if prompt:
        current_prompt_context.set(prompt)
    
    # set the current trace attributes
    if name:
        update_current_trace(name=name)
    if tags:
        update_current_trace(tags=tags)
    if metadata:
        update_current_trace(metadata=metadata)
    if user_id:
        update_current_trace(user_id=user_id)
    if thread_id:
        update_current_trace(thread_id=thread_id)
    

    yield current_trace
