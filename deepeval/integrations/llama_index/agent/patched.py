from deepeval.tracing.utils import with_metrics


try:
    from llama_index.core.agent.workflow.function_agent import FunctionAgent
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
    pass