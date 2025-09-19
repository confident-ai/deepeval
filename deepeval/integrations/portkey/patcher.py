import functools
from deepeval.tracing.context import current_span_context
from deepeval.tracing.tracing import Observer
from typing import Optional, List
from deepeval.metrics.base_metric import BaseMetric
from deepeval.prompt import Prompt

try:
    from portkey_ai.api_resources.apis.chat_complete import Completions
    from deepeval.integrations.portkey.utils import extract_llm_output_from_chat_completion_messages
    is_portkey_installed = True
except Exception:
    is_portkey_installed = False

def is_portkey_available():
    if not is_portkey_installed:
        raise ImportError("portkey-ai is not available. Please install it with `pip install portkey-ai`.")
    return True

def _safe_patch_portkey_chat_completion_completions():
    is_portkey_available()
    current_create = getattr(Completions, "create", None)
    if current_create is None:
        return

    if getattr(current_create, "_deepeval_patched", False):
        return

    original_create = current_create

    @functools.wraps(original_create)
    def new_create(
        *args,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        prompt: Prompt,
        **kwargs
    ):
        with Observer(
            span_type="llm",
            metrics=metrics,
            metric_collection=metric_collection,
            observe_kwargs={"model": kwargs.get("model", "unknown")},
            func_name="LLM",
        ) as observer:
            result = original_create(*args, **kwargs)
            observer.result = extract_llm_output_from_chat_completion_messages(result)
            _current_span_context= current_span_context.get()
            _current_span_context.input = kwargs.get("messages")
            _current_span_context.prompt = prompt

        return result

    new_create._deepeval_patched = True
    new_create._deepeval_original = original_create # preserve original reference
    Completions.create = new_create
    
def safe_instrument_all():
    is_portkey_available()
    _safe_patch_portkey_chat_completion_completions()
