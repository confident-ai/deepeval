from typing import Optional, List
from deepeval.metrics.base_metric import BaseMetric
from deepeval.prompt import Prompt

try: 
    from portkey_ai import Portkey
    from portkey_ai.api_resources.apis.chat_complete import Completions, ChatCompletion
    from deepeval.integrations.portkey.patcher import safe_instrument_all
    portkey_installed = True
except Exception:
    portkey_installed = False

def is_portkey_available():
    if not portkey_installed:
        raise ImportError("portkey-ai is not available. Please install it with `pip install portkey-ai`.")
    return True

class _DeepevalCompletions(Completions):
    def create(
        self, 
        *args,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        prompt: Optional[Prompt] = None,
        **kwargs
    ):
        return super().create(
            *args, 
            metrics=metrics, 
            metric_collection=metric_collection, 
            prompt=prompt,
            **kwargs
        )

class _DeepevalChatCompletion(ChatCompletion):
    completions: _DeepevalCompletions

class DeepevalPortkey(Portkey):
    chat: _DeepevalChatCompletion

    def __init__(
        self,
        *args,
        **kwargs
    ):
        safe_instrument_all()
        super().__init__(*args, **kwargs)