from deepeval.telemetry import capture_tracing_integration

try:
    from pydantic_ai.agent import Agent

    pydantic_ai_installed = True
except:
    pydantic_ai_installed = False


def is_pydantic_ai_installed():
    if not pydantic_ai_installed:
        raise ImportError(
            "Pydantic AI is not installed. Please install it with `pip install pydantic-ai`."
        )


class PydanticAIAgent(Agent):
    def __init__(
        self,
        *args,
        metric_collection: str = None,
        trace_attributes: dict = None,
        **kwargs
    ):
        with capture_tracing_integration("pydantic_ai.agent.PydanticAIAgent"):
            is_pydantic_ai_installed()
            super().__init__(*args, **kwargs)
            self.metric_collection = metric_collection
            self.trace_attributes = trace_attributes
