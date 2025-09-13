import functools
from typing import Optional, List
from deepeval.tracing.context import current_span_context
from deepeval.tracing.tracing import Observer
from deepeval.tracing.utils import make_json_serializable
from deepeval.metrics import BaseMetric
try:
    from portkey_ai import Portkey
    from portkey_ai.api_resources.apis.chat_complete import Completions, ChatCompletions
    is_portkey_installed = True
except Exception:
    is_portkey_installed = False

def is_portkey_available():
    if not is_portkey_installed:
        raise ImportError("portkey-ai is not available. Please install it with `pip install portkey-ai`.")
    return True

def _patch_portkey_init():
    original_init = Portkey.__init__

    @functools.wraps(original_init)
    def new_init(*args, metric_collection: Optional[str] = None, metrics: Optional[List[BaseMetric]] = None, **kwargs):
        result = original_init(*args, **kwargs)
        _patch_portkey_chat_completions(args[0].chat.completions, metrics, metric_collection)
        return result

    Portkey.__init__ = new_init


def _patch_portkey_chat_completions(completions: Completions, metrics=None, metric_collection=None):
    if getattr(completions, "_deepeval_patched", False):
        return

    original_create = completions.create  # capture bound original
    
    @functools.wraps(original_create)
    def new_create(*args, **kwargs):
        with Observer(
            span_type="llm",
            metrics=metrics,
            metric_collection=metric_collection,
            observe_kwargs={"model": kwargs.get("model", "unknown")},
            func_name="LLM",
        ) as observer:
            result = original_create(*args, **kwargs)
            observer.result = extract_chat_completion_messages(result)
            current_span_context.get().input = kwargs.get("messages")
        return result

    completions.create = new_create
    setattr(completions, "_deepeval_patched", True)

def instrument():
    is_portkey_available()
    _patch_portkey_init()

def extract_chat_completion_messages(result: ChatCompletions):
    try:
        choices = None
        if hasattr(result, "choices"):
            choices = result.choices
        elif isinstance(result, dict):
            choices = result.get("choices")

        messages = []
        if isinstance(choices, list):
            for c in choices:
                message = None
                if hasattr(c, "message"):
                    message = c.message
                elif isinstance(c, dict):
                    message = c.get("message")
                if message is not None:
                    messages.append(make_json_serializable(message))

        return messages if messages else make_json_serializable(result)
    except Exception:
        return make_json_serializable(result)