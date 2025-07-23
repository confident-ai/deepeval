from deepeval.tracing.utils import with_metrics


try:
    from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, CodeActAgent
    is_llama_index_installed = True
except:
    is_llama_index_installed = False

def is_llama_index_agent_installed():
    if not is_llama_index_installed:
        raise ImportError(
            "llama-index is neccesary for this functionality. Please install it with `pip install llama-index` or with package manager of choice."
        )


@with_metrics
class FunctionAgent(FunctionAgent):
    def __init__(self, *args, metric_collection=None, metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_collection = metric_collection
        self.metrics = metrics or []

@with_metrics
class ReActAgent(ReActAgent):
    def __init__(self, *args, metric_collection=None, metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_collection = metric_collection
        self.metrics = metrics or []

@with_metrics
class CodeActAgent(CodeActAgent):
    def __init__(self, *args, metric_collection=None, metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_collection = metric_collection
        self.metrics = metrics or []